// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_generate_dfg_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(GENERATE_DFG_THREAD_GROUP_SIZE, GENERATE_DFG_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5) / (float2)g_push_constants.m_dst_resolution;

    float l_nov = l_uv.x;
    float l_linear_roughness = l_uv.y;

    // Get normal vector and view vector
    const float3 l_n = float3(0, 0, 1);
    float3 l_v;
    l_v.x = sqrt(1.0f - l_nov * l_nov);
    l_v.y = 0.0f;
    l_v.z = l_nov;

    // Do integration
    l_rt[p_dispatch_thread_id.xy] = integrate_dfg(l_n, l_v, l_linear_roughness, g_push_constants.m_num_samples);
}
