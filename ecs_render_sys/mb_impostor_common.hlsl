// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MB_SHADER_IMPOSTOR_COMMON_H
#define MB_SHADER_IMPOSTOR_COMMON_H

#include "../helper_shaders/mb_common.hlsl"
#include "mb_hiz_culling.hlsl"

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
bool accept_impostor_instance(sb_impostor_item_t p_item, sb_impostor_instance_t p_instance, cb_camera_t p_camera, float p_impostor_distance, uint p_hiz_map_srv)
{
    // Discard impostors with no data
    if (p_item.m_albedo_alpha_srv == RAL_NULL_BINDLESS_INDEX || p_item.m_normal_depth_srv == RAL_NULL_BINDLESS_INDEX)
    {
        return false;
    }

    // Culling sphere vs frustum
    float4 l_bounding_sphere = float4(p_instance.m_position, p_item.m_bounding_sphere.w);
    if (!is_sphere_inside_camera_frustum(p_camera, l_bounding_sphere))
    {
        return false;
    }

    if (length(p_instance.m_position) < p_impostor_distance)
    {
        return false;
    }

    // Occlusion culling
    if (p_hiz_map_srv != RAL_NULL_BINDLESS_INDEX)
    {
        float3 l_aabb_min = p_item.m_bounding_sphere.xyz - p_item.m_bounding_sphere.www;
        float3 l_aabb_max = p_item.m_bounding_sphere.xyz + p_item.m_bounding_sphere.www;

        if (!hiz_visibility_test(p_camera,
                                 p_item.m_bounding_sphere,
                                 l_aabb_min,
                                 l_aabb_max,
                                 get_impostor_instance_transform(p_instance),
                                 p_hiz_map_srv))
        {
            return false;
        }

    }

    return true;
}

#endif //MB_SHADER_IMPOSTOR_COMMON_H