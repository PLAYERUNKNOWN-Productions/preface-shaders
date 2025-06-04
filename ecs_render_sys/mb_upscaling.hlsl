// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_upscaling_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(POSTPROCESS_GENERAL_THREAD_GROUP_SIZE, POSTPROCESS_GENERAL_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get remapped uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2)g_push_constants.m_dst_resolution;
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float4 l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv);

    l_rt[p_dispatch_thread_id.xy] = l_color;
}
