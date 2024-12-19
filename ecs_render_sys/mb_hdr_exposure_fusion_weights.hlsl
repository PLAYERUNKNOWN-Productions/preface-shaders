// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_postprocess_vs.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_exposure_fusion_weights_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------
float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_color = bindless_tex2d_sample_level(g_push_constants.m_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord, 0).rgb;

    // Compute the synthetic exposure weights.
    float3 l_diff = l_color - float3(0.5, 0.5, 0.5);
    float3 l_weights = exp(-0.5 * l_diff * l_diff * g_push_constants.m_sigma_squared);
    l_weights /= dot(l_weights, float3(1,1,1)) + 0.0001;

    return float4(l_weights, 1.0);
}
