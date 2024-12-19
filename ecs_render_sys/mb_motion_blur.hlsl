// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_postprocess_vs.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_motion_blur_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

float2 velocity_threshold(float2 l_velocity, float p_threshold, float p_max)
{
    float l_velocity_magnitude = length(l_velocity);
    float l_velocity_magnitude_clamped = clamp(l_velocity_magnitude - p_threshold, 0, p_max);
    l_velocity = l_velocity_magnitude > 0 ? normalize(l_velocity) * l_velocity_magnitude_clamped : 0;
    return l_velocity;
}

[numthreads(MOTION_BLUR_THREAD_GROUP_SIZE, MOTION_BLUR_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 l_dsc_dim = uint2(g_push_constants.m_dst_resolution_x, g_push_constants.m_dst_resolution_y);
    if (any(p_dispatch_thread_id.xy >= l_dsc_dim))
    {
        return;
    }

    RWTexture2D<float4> l_dst_uav = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];
    Texture2D<float2> l_velocity_srv = ResourceDescriptorHeap[g_push_constants.m_velocity_texture_srv];
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2)l_dsc_dim;

    // Get remapped uv
    float2 l_remapped_uv = get_remapped_uv(l_uv, l_camera.m_render_scale);

    // Get depth
    float l_depth = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState) SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_remapped_uv).r;

    // Get world space local position
    float3 l_pos_ws_local = get_world_space_local_position(l_uv, l_depth, l_camera.m_inv_view_proj_local);
    float4 l_proj_pos = mul(float4(l_pos_ws_local, 1.0f), l_camera.m_view_proj_local_prev);
    float2 l_uv_prev = (l_proj_pos.xy / l_proj_pos.w) * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);

    // Convert to UV-space
    float2 l_object_velocity = g_push_constants.m_object_velocity_scale * l_velocity_srv[p_dispatch_thread_id.xy] * float2(0.5f, -0.5f);
    l_object_velocity = velocity_threshold(l_object_velocity, g_push_constants.m_object_threshold_velocity, g_push_constants.m_object_max_velocity);

    // Compute camera velocity
    float2 l_camera_velocity = g_push_constants.m_camera_velocity_scale * (l_uv - l_uv_prev);
    l_camera_velocity = velocity_threshold(l_camera_velocity, g_push_constants.m_camera_threshold_velocity, g_push_constants.m_camera_max_velocity);

    // Add camera and object velocities
    float2 l_velocity = l_object_velocity + l_camera_velocity;

    // Clamp velocity
    float l_velocity_magnitude = length(l_velocity);

    // Vary sample count based on velocity amgnitude
    const uint c_min_sample_count = 2; // MUST be even number to maintain kernel balance
    const uint c_max_sample_count = 8; // MUST be even number to maintain kernel balance
    uint l_num_samples = clamp(l_velocity_magnitude * max(l_dsc_dim.x, l_dsc_dim.y), c_min_sample_count, c_max_sample_count);

    // Scale kernel based on the pass to improve filtering
    l_velocity *= g_push_constants.m_kernel_size_multiplier;

    // Decompose filtering into N passes
    // Each next pass is filtering "in-between" two samples of the previous pass
    // M passes N samples each is equal to one pass with N^M samples
    l_velocity *= 1.0f / pow(l_num_samples, (float)g_push_constants.m_pass_index);

    float3 l_color = 0;
    for (int l_sample_index = 0; l_sample_index < l_num_samples; ++l_sample_index)
    {
        float2 l_sample_uv = l_uv + l_velocity * (l_sample_index / (float) (l_num_samples - 1) - 0.5f); // Center samples around the source point
        l_color += bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_sample_uv).xyz;
    }

    l_dst_uav[p_dispatch_thread_id.xy] = float4(l_color / (float) l_num_samples, 1.0f);

    // Highlight moving objects
#if 0
    if (l_velocity_magnitude > 0.00001)
    {
        l_dst_uav[p_dispatch_thread_id.xy] = float4(1.0f, 0, 0, 1.0f);
    }
#endif
    //l_dst_uav[p_dispatch_thread_id.xy] = float4(length(l_uv - l_uv_prev), 0, 0, 1.0f);
}
