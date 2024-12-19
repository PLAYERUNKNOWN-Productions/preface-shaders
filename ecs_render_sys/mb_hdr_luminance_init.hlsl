// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_luminance_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(POSTPROCESS_GENERAL_THREAD_GROUP_SIZE, POSTPROCESS_GENERAL_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2)g_push_constants.m_dst_resolution;

    float3 l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv).rgb;
    l_color = unpack_lighting(l_color);

    // Get luminance from hdr color
    float l_lum = get_luminance(l_color);

#if LOG_LUMINANCE
    l_lum = log(l_lum + 1e-6);
#endif

    l_rt[p_dispatch_thread_id.xy] = l_lum;
}
