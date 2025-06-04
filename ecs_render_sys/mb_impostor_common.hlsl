// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_IMPOSTOR_COMMON_H
#define MB_SHADER_IMPOSTOR_COMMON_H

#include "../helper_shaders/mb_common.hlsl"
#include "mb_hiz_culling.hlsl"

float4x3 get_impostor_instance_transform(sb_impostor_instance_t p_instance)
{
    float l_sin = 0;
    float l_cos = 1;
    sincos(p_instance.m_angle, l_sin, l_cos);

    float3 l_normal   = p_instance.m_up_vector;
    float3 l_binormal = normalize(cross(float3(1,0,0), l_normal));
    float3 l_tangent  = normalize(cross(l_normal,      l_binormal));

    float4x3 l_combined_transform;
    l_combined_transform._11_12_13 = (l_cos * l_tangent - l_sin * l_binormal) * p_instance.m_scale;
    l_combined_transform._21_22_23 = l_normal * p_instance.m_scale;
    l_combined_transform._31_32_33 = (l_sin * l_tangent + l_cos * l_binormal) * p_instance.m_scale;
    l_combined_transform._41_42_43 = p_instance.m_position;
    return l_combined_transform;
}

bool accept_impostor_instance(sb_impostor_item_t p_item,
                              sb_impostor_instance_t p_instance,
                              cb_camera_t p_camera,
                              uint p_hiz_map_srv,
                              float p_lod_bias)
{
    // Discard impostors with no data
    if (p_item.m_albedo_alpha_srv == RAL_NULL_BINDLESS_INDEX || p_item.m_normal_depth_srv == RAL_NULL_BINDLESS_INDEX)
    {
        return false;
    }

    // Impostors are generated from population, which always has custom culling scale set to zero
    float l_render_instance_culling_scale = 0;
    float4x3 l_transform = get_impostor_instance_transform(p_instance);

    // Culling sphere vs frustum
    float4 l_bounding_sphere = get_bounding_sphere_ws(p_item.m_bounding_sphere, l_transform, l_render_instance_culling_scale);
    if (!is_sphere_inside_camera_frustum(p_camera, l_bounding_sphere))
    {
        return false;
    }

    // Screen coverage test
    float l_distance_to_lod_camera = length(p_instance.m_position) * p_lod_bias;
    if (!screen_coverage_test(p_camera.m_proj, p_camera.m_fov_vertical, l_bounding_sphere, l_distance_to_lod_camera, p_item.m_screen_coverage_range))
    {
        return false;
    }

    // Occlusion culling
    if (p_hiz_map_srv != RAL_NULL_BINDLESS_INDEX)
    {
        // Although impostors use a custom culling scale of 0, we cannot use 0 here as it would set the radius of our model space bounding sphere to 0
        float4 l_bounding_sphere_ms = get_bounding_sphere_ms(p_item.m_bounding_sphere, 1);
        float3 l_aabb_min = l_bounding_sphere_ms.xyz - l_bounding_sphere_ms.www;
        float3 l_aabb_max = l_bounding_sphere_ms.xyz + l_bounding_sphere_ms.www;

        if (!hiz_visibility_test(p_camera, l_aabb_min, l_aabb_max, l_transform, p_hiz_map_srv))
        {
            return false;
        }

    }

    return true;
}

#endif //MB_SHADER_IMPOSTOR_COMMON_H
