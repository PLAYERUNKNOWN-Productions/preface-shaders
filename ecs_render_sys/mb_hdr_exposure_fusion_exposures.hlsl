// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_hdr_tone_mapping_common.hlsl"
#include "mb_postprocess_vs.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_exposure_fusion_exposures_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------
float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_color = bindless_tex2d_sample_level(g_push_constants.m_hdr_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord, 0).rgb;
    l_color = unpack_lighting(l_color);

    // Get exposure from eye adaptation result
    float l_lum = bindless_tex2d_sample_level(g_push_constants.m_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(g_push_constants.m_lum_uvx, 0.5f), 0).r;
    float l_exposure_scale = g_push_constants.m_key_value / l_lum;

    // Apply exposure scale
    l_color.rgb *= l_exposure_scale;

    // Tonemap three syntetic exposures and produce their luminances.
    float l_highlights = linear_to_gamma(get_luminance(tonemap_aces_fitted(l_color * g_push_constants.m_exposure_highlights)));
    float l_midtones   = linear_to_gamma(get_luminance(tonemap_aces_fitted(l_color)                                         ));
    float l_shadows    = linear_to_gamma(get_luminance(tonemap_aces_fitted(l_color * g_push_constants.m_exposure_shadows   )));

    return float4(l_highlights, l_midtones, l_shadows, 1.0);
}
