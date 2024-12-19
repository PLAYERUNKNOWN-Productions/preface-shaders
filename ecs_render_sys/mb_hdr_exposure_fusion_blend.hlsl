// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_postprocess_vs.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_exposure_fusion_blend_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------
float ps_main(ps_input_t p_input) : SV_TARGET
{
    // Blend the exposures based on the blend weights.
    float3 l_weights   = bindless_tex2d_sample(g_push_constants.m_weights_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;
    float3 l_exposures = bindless_tex2d_sample(g_push_constants.m_exposures_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;
    l_weights /= dot(l_weights, float3(1,1,1)) + 0.0001;
    return dot(l_exposures * l_weights, float3(1,1,1));
}
