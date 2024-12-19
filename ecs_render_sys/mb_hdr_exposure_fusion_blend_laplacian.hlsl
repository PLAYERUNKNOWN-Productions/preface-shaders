// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_postprocess_vs.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_exposure_fusion_blend_laplacian_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------
float ps_main(ps_input_t p_input) : SV_TARGET
{
    // Blend the Laplacians based on exposure weights.
    float  l_accum            = bindless_tex2d_sample(g_push_constants.m_accum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).r;
    float3 l_exposure         = bindless_tex2d_sample(g_push_constants.m_exposures_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;
    float3 l_exposure_coarser = bindless_tex2d_sample(g_push_constants.m_exposures_coarser_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;
    float3 l_weights          = bindless_tex2d_sample(g_push_constants.m_weights_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;

    float3 l_laplacians = l_exposure - l_exposure_coarser;
    l_weights *= g_push_constants.m_boost_local_constrast > 0 ? abs(l_laplacians) + 0.0001 : float3(1,1,1);
    l_weights /= dot(l_weights, float3(1,1,1)) + 0.0001;
    float l_laplac = dot(l_laplacians * l_weights, float3(1,1,1));
    return (l_accum + l_laplac);
}
