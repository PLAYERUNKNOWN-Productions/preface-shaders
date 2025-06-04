// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_HIZ_CULLING_H
#define MB_SHADER_HIZ_CULLING_H

//#define BUILD_AABB_FROM_BOUNDING_SPHERE
#define USE_SAMPLER

#ifdef OCCLUSION_CULL_DEBUG_DRAW
#include "../helper_shaders/mb_debug_render.hlsl"
#endif

#include "../helper_shaders/mb_common.hlsl"

#ifdef OCCLUSION_CULL_DEBUG_DRAW
static float3 s_color_table[12] =
{
    float3(0.0f, 0.0f, 1.0f),   //0  Blue
    float3(1.0f, 1.0f, 0.0f),   //1  Yellow
    float3(1.0f, 0.0f, 1.0f),   //2  Magenta
    float3(0.0f, 1.0f, 1.0f),   //3  Cyan
    float3(1.0f, 0.5f, 0.0f),   //4  Orange
    float3(0.5f, 0.0f, 1.0f),   //5  Purple
    float3(0.0f, 0.5f, 0.5f),   //6  Teal
    float3(0.0f, 0.5f, 0.5f),   //7  Medium Aquamarine
    float3(0.2f, 0.5f, 0.5f),   //8  Steel Blue
    float3(0.5f, 0.2f, 0.0f),   //9  Brown
    float3(0.0f, 0.0f, 0.5f),   //10 Navy
    float3(0.0f, 0.5f, 0.0f),   //11 Olive
    //
};
#endif

bool is_closer(float p_left, float p_right)
{
    return p_left > p_right;
}

float furthest(float p_left, float p_right)
{
    return min(p_left, p_right);
}

float closest(float p_left, float p_right)
{
    return max(p_left, p_right);
}

bool hiz_visibility_test(cb_camera_t p_camera, float3 p_aabb_min_ms, float3 p_aabb_max_ms, float4x3 p_instance_transform, uint p_hiz_map_srv)
{
    float3 l_min_uv = float3(1, 1, 1);
    float3 l_max_uv = float3(0, 0, 0);

    static const int l_point_count = 8;

    //build AABB
    float3 l_min = p_aabb_min_ms;
    float3 l_max = p_aabb_max_ms;

    float3 l_aabb_local_space[l_point_count] =
    {
        float3(l_min.x, l_max.y, l_min.z), //front top left
        float3(l_max.x, l_max.y, l_min.z), //front top right
        float3(l_min.x, l_min.y, l_min.z), //front bot left
        float3(l_max.x, l_min.y, l_min.z), //front bot right
        float3(l_min.x, l_max.y, l_max.z), //back top left
        float3(l_max.x, l_max.y, l_max.z), //back top right
        float3(l_min.x, l_min.y, l_max.z), //back bot left
        float3(l_max.x, l_min.y, l_max.z), //back bot right
    };

    float4 l_minimum_depth_view_space = float4(0, 0, 15, 1);  //todo make this 15 tweakable
    float4 l_minimum_depth_projection_space = mul(l_minimum_depth_view_space, p_camera.m_proj);
    float l_minimum_depth_ndc = l_minimum_depth_projection_space.z / l_minimum_depth_projection_space.w;

    float4x4 l_transform =
    {
        p_instance_transform[0], 0,
        p_instance_transform[1], 0,
        p_instance_transform[2], 0,
        p_instance_transform[3], 1,
    };

    float4x4 l_mvp = mul(mul(l_transform, p_camera.m_view_local), p_camera.m_proj);
    for(uint l_i = 0; l_i < l_point_count; ++l_i)
    {
        float3 l_point_uv = pos_to_uv_depth(l_aabb_local_space[l_i], l_mvp);

        l_min_uv = min(l_point_uv, l_min_uv);
        l_max_uv = max(l_point_uv, l_max_uv);
    }

    l_min_uv = saturate(l_min_uv);
    l_max_uv = saturate(l_max_uv);

    float l_closest_depth = closest(l_min_uv.z, l_max_uv.z);
    if(is_closer(l_closest_depth, l_minimum_depth_ndc))
    {
          return true;
    }

#ifdef OCCLUSION_CULL_DEBUG_DRAW
    //bounding box

    float2 l_top_left       = float2(l_min_uv.x, l_min_uv.y);
    float2 l_top_right      = float2(l_max_uv.x, l_min_uv.y);
    float2 l_bottom_left    = float2(l_min_uv.x, l_max_uv.y);
    float2 l_bottom_right   = float2(l_max_uv.x, l_max_uv.y);

    float3 l_top_left_view_space = uv_depth_to_pos(l_top_left, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_top_right_view_space = uv_depth_to_pos(l_top_right, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_bottom_left_view_space = uv_depth_to_pos(l_bottom_left, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_bottom_right_view_space = uv_depth_to_pos(l_bottom_right, 1.f, p_camera.m_inv_view_proj_local);
#endif

    float2 l_delta_uv = l_max_uv.xy - l_min_uv.xy;
    float2 l_center_uv = l_min_uv.xy + l_delta_uv * .5f;

    Texture2D<float> l_hiz_map = ResourceDescriptorHeap[p_hiz_map_srv];

    float2 l_hiz_dimensions;
    l_hiz_map.GetDimensions(l_hiz_dimensions.x, l_hiz_dimensions.y);

    float2 l_bounding_box_dimensions = l_delta_uv * l_hiz_dimensions;

    float l_dist = max(l_bounding_box_dimensions.x, l_bounding_box_dimensions.y);
    uint l_mip_level = floor(log2(l_dist));

    float2 l_hiz_mip_dimensions;
    uint l_num_mips;
    l_hiz_map.GetDimensions(l_mip_level, l_hiz_mip_dimensions.x, l_hiz_mip_dimensions.y, l_num_mips);

#ifdef OCCLUSION_CULL_DEBUG_DRAW
    //boundig box center

    uint2 l_center_pixel_top_left       = l_center_uv.xy * l_hiz_dimensions;
    uint2 l_center_pixel_top_right      = l_center_pixel_top_left + uint2(1, 0);
    uint2 l_center_pixel_bottom_left    = l_center_pixel_top_left + uint2(0, 1);
    uint2 l_center_pixel_bottom_right   = l_center_pixel_top_left + uint2(1, 1);

    float2 l_center_pixel_top_left_uv = float2(l_center_pixel_top_left) / l_hiz_dimensions;
    float2 l_center_pixel_top_right_uv = float2(l_center_pixel_top_right) / l_hiz_dimensions;
    float2 l_center_pixel_bottom_left_uv = float2(l_center_pixel_bottom_left) / l_hiz_dimensions;
    float2 l_center_pixel_bottom_right_uv = float2(l_center_pixel_bottom_right) / l_hiz_dimensions;

    float3 l_center_pixel_top_left_view_space = uv_depth_to_pos(l_center_pixel_top_left_uv, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_center_pixel_top_right_view_space = uv_depth_to_pos(l_center_pixel_top_right_uv, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_center_pixel_bottom_left_view_space = uv_depth_to_pos(l_center_pixel_bottom_left_uv, 1.f, p_camera.m_inv_view_proj_local);
    float3 l_center_pixel_bottom_right_view_space = uv_depth_to_pos(l_center_pixel_bottom_right_uv, 1.f, p_camera.m_inv_view_proj_local);
#endif

#ifdef USE_SAMPLER
    float l_furthest_sample = l_hiz_map.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_center_uv.xy, l_mip_level);
#else
    uint3 l_coords0 = uint3(l_center_uv.xy * l_hiz_mip_dimensions, l_mip_level);
    uint3 l_coords1 = l_coords0 + uint3(1, 0, 0);
    uint3 l_coords2 = l_coords0 + uint3(0, 1, 0);
    uint3 l_coords3 = l_coords0 + uint3(1, 1, 0);

    //we must prevent loading out of bounds since this will return 0, effectivly making things never occluded
    l_coords0.xy = clamp(l_coords0.xy, uint2(0,0), uint2(l_hiz_mip_dimensions));
    l_coords1.xy = clamp(l_coords1.xy, uint2(0,0), uint2(l_hiz_mip_dimensions));
    l_coords2.xy = clamp(l_coords2.xy, uint2(0,0), uint2(l_hiz_mip_dimensions));
    l_coords3.xy = clamp(l_coords3.xy, uint2(0,0), uint2(l_hiz_mip_dimensions));

    float l_sample0 = l_hiz_map.Load(l_coords0);
    float l_sample1 = l_hiz_map.Load(l_coords1);
    float l_sample2 = l_hiz_map.Load(l_coords2);
    float l_sample3 = l_hiz_map.Load(l_coords3);

    float l_furthest_sample = furthest(furthest(l_sample0, l_sample1), furthest(l_sample2, l_sample3));
#endif

    bool l_visible = is_closer(l_closest_depth, l_furthest_sample);

#ifdef OCCLUSION_CULL_DEBUG_DRAW

    float3 l_color = l_visible ? s_color_table[l_mip_level] : float3(1.f, 0.f, 0.f);

    draw_line(l_top_left_view_space,        l_top_right_view_space,     l_color);
    draw_line(l_top_right_view_space,       l_bottom_right_view_space,  l_color);
    draw_line(l_bottom_right_view_space,    l_bottom_left_view_space ,  l_color);
    draw_line(l_bottom_left_view_space,    l_top_left_view_space,       l_color);

    draw_line(l_center_pixel_top_left_view_space,       l_center_pixel_top_right_view_space,    float3(1.f, 0.f, 0.f));
    draw_line(l_center_pixel_top_right_view_space,      l_center_pixel_bottom_right_view_space, float3(1.f, 0.f, 0.f));
    draw_line(l_center_pixel_bottom_right_view_space,   l_center_pixel_bottom_left_view_space,  float3(1.f, 0.f, 0.f));
    draw_line(l_center_pixel_bottom_left_view_space,    l_center_pixel_top_left_view_space,     float3(1.f, 0.f, 0.f));

#endif

    return l_visible;
}

#endif //MB_SHADER_HIZ_CULLING_H
