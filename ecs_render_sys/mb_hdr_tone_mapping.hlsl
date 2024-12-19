// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_hdr_tone_mapping_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
    float4 m_position : SV_POSITION;
    float2 m_texcoord : TEXCOORD0;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

#if EYE_ADAPTATION
ConstantBuffer<cb_push_tone_mapping_eye_adaptation_t>   g_push_constants    : register(REGISTER_PUSH_CONSTANTS);
#else // fixed exposure
ConstantBuffer<cb_push_tone_mapping_fixed_exposure_t>   g_push_constants    : register(REGISTER_PUSH_CONSTANTS);
#endif

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float4 l_color = bindless_tex2d_sample_level(g_push_constants.m_hdr_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_input.m_texcoord, 0);
    l_color.rgb = unpack_lighting(l_color.rgb);

#if EYE_ADAPTATION
    // Get exposure from eye adaptation result
    float l_lum = bindless_tex2d_sample_level(g_push_constants.m_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(g_push_constants.m_lum_uvx, 0.5f), 0).r;
    float l_exposure_scale = g_push_constants.m_key_value / l_lum;

    // Apply exposure scale
    l_color.rgb *= l_exposure_scale;
#else // fixed exposure
    float l_exposure_scale = ev100_to_exposure(g_push_constants.m_key_value);
    l_color.rgb *= l_exposure_scale;
#endif

    // Tone mapping
    if (g_push_constants.m_tonemap_operator == TONEMAP_OP_EXP)
    {
        l_color.rgb = tonemap_exp(l_color.rgb);
    }
    else if (g_push_constants.m_tonemap_operator == TONEMAP_OP_REINHARD)
    {
        l_color.rgb = tonemap_reinhard(l_color.rgb);
    }
    else if (g_push_constants.m_tonemap_operator == TONEMAP_OP_REINHARD_JODIE)
    {
        l_color.rgb = tonemap_reinhard_jodie(l_color.rgb);
    }
    else if (g_push_constants.m_tonemap_operator == TONEMAP_OP_UNCHARTED2_FILMIC)
    {
        l_color.rgb = tonemap_uncharted2_filmic(l_color.rgb);
    }
    else if (g_push_constants.m_tonemap_operator == TONEMAP_OP_ACES_APPROX)
    {
        l_color.rgb = tonemap_aces_approx(l_color.rgb);
    }
    else if (g_push_constants.m_tonemap_operator == TONEMAP_OP_ACES_FITTED)
    {
        l_color.rgb = tonemap_aces_fitted(l_color.rgb);
    }
    
	l_color.rgb = linear_to_gamma(l_color.rgb);
	
    return l_color;
}
