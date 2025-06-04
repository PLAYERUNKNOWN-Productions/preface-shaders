// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// CBV
ConstantBuffer<cb_push_lighting_combination_t>  g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

[numthreads(LIGHTING_COMBINATION_THREAD_GROUP_SIZE, LIGHTING_COMBINATION_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_scene_rt_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2)g_push_constants.m_dst_resolution;

    // Get direct lighting & opacity
    float4 l_direct_lighting_data = bindless_tex2d_sample_level(g_push_constants.m_direct_lighting_rt_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv);
    float3 l_direct_lighting = unpack_lighting(l_direct_lighting_data.rgb);
    float l_opacity = l_direct_lighting_data.a;

    // Get indirect lighting
    float3 l_indirect_lighting = unpack_lighting(bindless_tex2d_sample_level(g_push_constants.m_indirect_lighting_rt_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).rgb);

#if SSAO // Apply screen space ambient occlusion
    const float l_ssao = bindless_tex2d_sample_level(g_push_constants.m_ssao_rt_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).r;
    l_indirect_lighting *= l_ssao;
#endif

    // Get final lighting
    float3 l_final_lighting = l_direct_lighting + l_indirect_lighting;

    // Store result
    l_rt[p_dispatch_thread_id.xy] = float4(pack_lighting(l_final_lighting), l_opacity);
}
