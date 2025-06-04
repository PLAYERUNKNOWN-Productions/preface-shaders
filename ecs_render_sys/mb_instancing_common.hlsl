// Copyright:   PlayerUnknown Productions BV

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

// Skip object with small coverage values
bool skip_instances_with_small_coverage(in cb_camera_t p_camera, in float4 p_bounding_sphere, in float p_distance_z)
{
    // Different logic depending on orthographic or perspective camera
    bool l_is_orthographic = p_camera.m_proj._44 == 1.0f;
    float l_screen_coverage_threshold = l_is_orthographic ? 20.0f / 4096.0f : 10.0f / 4096.0f;
    float l_screen_coverage = l_is_orthographic ? p_bounding_sphere.w * max(p_camera.m_proj._11, p_camera.m_proj._22) : p_bounding_sphere.w / (p_distance_z * p_camera.m_fov_vertical);
    return l_screen_coverage > l_screen_coverage_threshold;
}

bool accept_render_item(sb_render_item_t p_render_item,
                        float4x3 p_render_instance_transform,
                        float p_render_instance_culling_scale,
                        cb_camera_t p_camera,
                        cb_camera_t p_lod_camera,
                        float3 p_lod_camera_offset,
                        uint p_hiz_map_srv,
                        float p_lod_bias,
                        uint p_forced_lod,
                        bool p_is_terrain)
{
    //Early out if this camera and this render item have no overlapping passes
    if ((p_render_item.m_render_output_mask & p_camera.m_render_output_mask) == 0)
    {
        return false;
    }

    // Transform bounding sphere to world space(camera-local)
    float4 l_bounding_sphere = get_bounding_sphere_ws(p_render_item.m_bounding_sphere_ms, p_render_instance_transform, p_render_instance_culling_scale);

    // Culling
    if (!is_sphere_inside_camera_frustum(p_camera, l_bounding_sphere))
    {
        return false;
    }

    // Forcing a LOD level overrides coverage tests, because coverage is the tool we use to select LODs
#ifdef MB_FORCE_LOD_LEVEL
    if (!p_is_terrain && p_render_item.m_lod_index != p_forced_lod)
    {
        return false;
    }
#else

#ifdef MB_ENABLE_SMALL_COVERAGE_TEST
    // Skip instances with small screen coverage
    float l_distance = length(p_render_instance_transform._41_42_43);
    if (!skip_instances_with_small_coverage(p_camera, l_bounding_sphere, l_distance))
    {
        return false;
    }
#endif

    // Screen coverage test
    float l_distance_to_lod_camera = length(p_render_instance_transform._41_42_43 - p_lod_camera_offset) * p_lod_bias;
    if (!screen_coverage_test(p_lod_camera.m_proj, p_lod_camera.m_fov_vertical, l_bounding_sphere - float4(p_lod_camera_offset, 0), l_distance_to_lod_camera, p_render_item.m_screen_coverage_range))
    {
        return false;
    }

#endif // MB_FORCE_LOD_LEVEL

#ifdef MB_HIZ_TEST
    if (p_hiz_map_srv != RAL_NULL_BINDLESS_INDEX)
    {
        float3 l_aabb_min = p_render_item.m_aabb_min_ms;
        float3 l_aabb_max = p_render_item.m_aabb_max_ms;

        if(p_is_terrain)
        {
            float4 l_bounding_sphere_ms = get_bounding_sphere_ms(p_render_item.m_bounding_sphere_ms, p_render_instance_culling_scale);

            l_aabb_min = l_bounding_sphere_ms.xyz - l_bounding_sphere_ms.www;
            l_aabb_max = l_bounding_sphere_ms.xyz + l_bounding_sphere_ms.www;
        }

        //hi-z occlusion cull test
        if (!hiz_visibility_test(p_camera,
                                 l_aabb_min,
                                 l_aabb_max,
                                 p_render_instance_transform,
                                 p_hiz_map_srv))
        {
            return false;
        }
    }
#endif

    return true;
}

#endif //MB_SHADER_INSTANCING_COMMON_H
