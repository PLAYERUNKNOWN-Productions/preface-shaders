// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_downsample_2x2_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(DOWNSAMPLE_THREAD_GROUP_SIZE, DOWNSAMPLE_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    const uint2 l_pos = p_dispatch_thread_id.xy;
    const float2 l_base_uv = float2(2.0f * l_pos.x * g_push_constants.m_inv_src_resolution.x, 2.0f * l_pos.y * g_push_constants.m_inv_src_resolution.y); // Upper left corner of the sampling area

    // Check the number of steps needed for the sampling loop.  *0 means can't be precomputed
    uint2 l_step;
    l_step.x = (g_push_constants.m_precomputed_step.x != 0) ? g_push_constants.m_precomputed_step.x : ((l_pos.x == g_push_constants.m_dst_resolution.x - 1) ? 3 : 2);
    l_step.y = (g_push_constants.m_precomputed_step.y != 0) ? g_push_constants.m_precomputed_step.y : ((l_pos.y == g_push_constants.m_dst_resolution.y - 1) ? 3 : 2);

    float2 l_uv = 0;
    float4 l_color = 0;

    // If 2x2, take advantage of the bilinear filter
    if (all(l_step == 2))
    {
        l_uv = l_base_uv + float2(g_push_constants.m_inv_src_resolution.x, g_push_constants.m_inv_src_resolution.y);
        l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv);
    }
    else // Otherwise, downsample manually
    {
        l_uv.x = l_base_uv.x + g_push_constants.m_inv_src_resolution.x * 0.5f;
        [unroll(9)]
        for (int l_i = 0; l_i < l_step.x; l_i++, l_uv.x += g_push_constants.m_inv_src_resolution.x)
        {
            l_uv.y = l_base_uv.y + g_push_constants.m_inv_src_resolution.y * 0.5f;
            for (int l_j = 0; l_j < l_step.y; l_j++, l_uv.y += g_push_constants.m_inv_src_resolution.y)
            {
                l_color += bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv);
            }
        }
        l_color /= (float)(l_step.x * l_step.y);
    }

    l_rt[p_dispatch_thread_id.xy] = l_color;
}
