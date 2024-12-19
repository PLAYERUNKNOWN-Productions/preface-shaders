// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MB_SHADER_INSTANCING_COMMON_H
#define MB_SHADER_INSTANCING_COMMON_H

#ifdef OCCLUSION_CULL_DEBUG_DRAW   
#include "../helper_shaders/mb_debug_render.hlsl"
#endif

#include "../helper_shaders/mb_common.hlsl"
#include "mb_hiz_culling.hlsl"

sb_render_instance_t get_instance_with_conversion(uint p_instance_buffer_srv, uint p_instance_index)
{
#ifdef MB_USE_POPULATION_INSTANCES
    // Fill a regular instance with a population instance
    StructuredBuffer<sb_render_instance_population_t> l_render_instance_buffer = ResourceDescriptorHeap[p_instance_buffer_srv];
    sb_render_instance_population_t l_render_instance_population = l_render_instance_buffer[p_instance_index];

    // Get rotation sin/cos
    float l_cos;
    float l_sin;
    sincos(l_render_instance_population.m_rotation, l_sin, l_cos);

    // Get spherical shape's TBN frame
    float3 l_normal = l_render_instance_population.m_normal;
    float3 l_binormal = normalize(cross(float3(1.0, 0, 0), l_normal));
    float3 l_tangent = normalize(cross(l_normal, l_binormal));

    // Build combined transform
    float4x3 l_combined_transform;
    l_combined_transform._11_12_13 = (l_cos * l_tangent - l_sin * l_binormal) * l_render_instance_population.m_scale;
    l_combined_transform._21_22_23 = l_normal * l_render_instance_population.m_scale;
    l_combined_transform._31_32_33 = (l_sin * l_tangent + l_cos * l_binormal) * l_render_instance_population.m_scale;
    l_combined_transform._41_42_43 = l_render_instance_population.m_position;

    sb_render_instance_t l_render_instance = (sb_render_instance_t)0;

    l_render_instance.m_transform       = l_combined_transform;
    l_render_instance.m_transform_prev  = l_combined_transform;
    l_render_instance.m_render_item_idx = l_render_instance_population.m_render_item_idx;
    l_render_instance.m_entity_id       = l_render_instance_population.m_entity_id;
    l_render_instance.m_user_data       = INSTANCING_NO_USER_DATA;

    return l_render_instance;
#else
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[p_instance_buffer_srv];
    return l_render_instance_buffer[p_instance_index];
#endif
}

//-----------------------------------------------------------------------------
// Skip object with small coverage values
bool skip_instances_with_small_coverage(in cb_camera_t p_camera, in float4 p_bounding_sphere, in float p_distance_z)
{
    // Different logic depending on ortho or proj camera
    bool l_is_orthographic = p_camera.m_proj._44 == 1.0f;
    float l_screen_coverage_threshold = l_is_orthographic ? 20.0f / 4096.0f : 10.0f / 4096.0f;
    float l_screen_coverage = l_is_orthographic ? p_bounding_sphere.w * max(p_camera.m_proj._11, p_camera.m_proj._22) : p_bounding_sphere.w / (p_distance_z * p_camera.m_fov_vertical.y);
    return l_screen_coverage > l_screen_coverage_threshold;
}

//-----------------------------------------------------------------------------
// Bounding sphere vs frustum
bool screen_coverage_test(in cb_camera_t p_camera, in float4 p_bounding_sphere, in float p_distance_z, in float2 p_threshold_range)
{
#ifdef MB_FORCE_LAST_LOD
    // HACK: We don't have a simple way to provide this value easily due to how LODs work in Melba
    const float LAST_LOD_COVERAGE = 0.01;
    return p_threshold_range.x < LAST_LOD_COVERAGE && LAST_LOD_COVERAGE <= p_threshold_range.y;
#else
    // Different logic depending on ortho or proj camera
    float l_screen_coverage = p_camera.m_proj._44 == 1.0f ? p_bounding_sphere.w * max(p_camera.m_proj._11, p_camera.m_proj._22) : p_bounding_sphere.w / (p_distance_z * p_camera.m_fov_vertical.y);
    return p_threshold_range.x < l_screen_coverage && l_screen_coverage <= p_threshold_range.y;
#endif
}

//-----------------------------------------------------------------------------
// Bounding sphere vs frustum
float4 get_bounding_sphere_ws(float4 p_bounding_sphere_model_space, sb_render_instance_t p_render_instance)
{
    float3 l_bounding_sphere_center = mul(p_bounding_sphere_model_space.xyz, (float3x3)p_render_instance.m_transform) + p_render_instance.m_transform._41_42_43;
    float3 l_transform_scale = float3(length(p_render_instance.m_transform._11_12_13), length(p_render_instance.m_transform._21_22_23), length(p_render_instance.m_transform._31_32_33));
    float l_transform_max_scale = max(l_transform_scale.x, max(l_transform_scale.y, l_transform_scale.z));
    l_transform_max_scale = max(l_transform_max_scale, p_render_instance.m_custom_culling_scale);
    float4 l_bounding_sphere = float4(l_bounding_sphere_center, l_transform_max_scale * p_bounding_sphere_model_space.w);
    return l_bounding_sphere;
}

//-----------------------------------------------------------------------------
bool accept_render_item(sb_render_item_t p_render_item,
                        sb_render_instance_t p_render_instance,
                        cb_camera_t p_camera,
                        cb_camera_t p_lod_camera,
                        float3 p_lod_camera_offset,
                        float p_impostor_distance,
                        uint p_hiz_map_srv,
                        float p_lod_bias)
{
    //Early out if this camera and this render item have no overlapping passes
    if ((p_render_item.m_render_output_mask & p_camera.m_render_output_mask) == 0)
    {
        return false;
    }

    // Transform bounding sphere to world space(camera-local)
    float4 l_bounding_sphere = get_bounding_sphere_ws(p_render_item.m_bounding_sphere_ms, p_render_instance);

    // Culling
    if (!is_sphere_inside_camera_frustum(p_camera, l_bounding_sphere))
    {
        return false;
    }

    // TODO: Share this with C++ code definition in mb_ecs_basic_comp.hpp
    static const uint l_camera_render_output_shadows      = 1u;
    static const uint l_camera_render_output_mask_shadows = 1u << l_camera_render_output_shadows;

    // Skip instances too close to the camera, allow other systems (like impostors) to be used there instead
    float l_distance = length(p_render_instance.m_transform._41_42_43);
    if (p_render_instance.m_render_item_idx != 0 &&                                     // Don't cull out terrain 
        (p_camera.m_render_output_mask & l_camera_render_output_mask_shadows) == 0 &&   // Don't cull geometry on the shadow pass
        l_distance > p_impostor_distance)
    {
        return false;
    }

#ifdef MB_ENABLE_SMALL_COVERAGE_TEST
    // Skip instances with small screen coverage
    if (!skip_instances_with_small_coverage(p_camera, l_bounding_sphere, l_distance))
    {
        return false;
    }
#endif

    // Screen coverage test
    float l_distance_to_lod_camera = length(p_render_instance.m_transform._41_42_43 - p_lod_camera_offset) * p_lod_bias;
    if (!screen_coverage_test(p_lod_camera, l_bounding_sphere - float4(p_lod_camera_offset, 0), l_distance_to_lod_camera, p_render_item.m_screen_coverage_range))
    {
        return false;
    }

#ifdef HIZ_TEST
    if (p_hiz_map_srv != RAL_NULL_BINDLESS_INDEX)
    {
        //hi-z occlusion cull test
        if (!hiz_visibility_test(p_camera,
                                 p_render_item.m_bounding_sphere_ms,
                                 p_render_item.m_aabb_min_ms,
                                 p_render_item.m_aabb_max_ms,
                                 p_render_instance.m_transform,
                                 p_hiz_map_srv))
        {
            return false;
        }
    }
#endif

    return true;
}

#endif //MB_SHADER_INSTANCING_COMMON_H