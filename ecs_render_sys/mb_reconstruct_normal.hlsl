// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// Push constants
ConstantBuffer<cb_push_reconstruct_normal_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 min_diff(float3 p_position, float3 p_position_right, float3 p_position_left)
{
    float3 l_v1 = p_position_right - p_position;
    float3 l_v2 = p_position - p_position_left;
    return (dot(l_v1, l_v1) < dot(l_v2, l_v2)) ? l_v1 : l_v2;
}

[numthreads(RECONSTRUCT_NORMAL_THREAD_GROUP_SIZE, RECONSTRUCT_NORMAL_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    // Output
    RWTexture2D<float4> l_normal_rt = ResourceDescriptorHeap[g_push_constants.m_dst_normal_texture_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2) g_push_constants.m_dst_resolution;

    float2 l_uv0 = l_uv;
    float2 l_uv1 = l_uv + float2( g_push_constants.m_inv_dst_resolution.x, 0); // right
    float2 l_uv2 = l_uv + float2(0,  g_push_constants.m_inv_dst_resolution.y); // top
    float2 l_uv3 = l_uv + float2(-g_push_constants.m_inv_dst_resolution.x, 0); // left
    float2 l_uv4 = l_uv + float2(0, -g_push_constants.m_inv_dst_resolution.y); // bottom

    // Get depth
    float l_depth0 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv0).r;
    float l_depth1 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv1).r;
    float l_depth2 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv2).r;
    float l_depth3 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv3).r;
    float l_depth4 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv4).r;

    // Get position
    float2 l_tan_half_fov_xy = get_tangent_half_fov_from_projection_matrix(l_camera.m_proj);
    float3 l_p0 = get_view_position(l_uv0, get_view_depth_from_depth(l_depth0, l_camera.m_z_near, l_camera.m_z_far), l_tan_half_fov_xy);
    float3 l_p1 = get_view_position(l_uv1, get_view_depth_from_depth(l_depth1, l_camera.m_z_near, l_camera.m_z_far), l_tan_half_fov_xy);
    float3 l_p2 = get_view_position(l_uv2, get_view_depth_from_depth(l_depth2, l_camera.m_z_near, l_camera.m_z_far), l_tan_half_fov_xy);
    float3 l_p3 = get_view_position(l_uv3, get_view_depth_from_depth(l_depth3, l_camera.m_z_near, l_camera.m_z_far), l_tan_half_fov_xy);
    float3 l_p4 = get_view_position(l_uv4, get_view_depth_from_depth(l_depth4, l_camera.m_z_near, l_camera.m_z_far), l_tan_half_fov_xy);

    float3 l_normal = normalize(cross(min_diff(l_p0, l_p1, l_p3), min_diff(l_p0, l_p2, l_p4)));
    l_normal_rt[p_dispatch_thread_id.xy] = float4(l_normal * 0.5 + 0.5, 0.0f);
}
