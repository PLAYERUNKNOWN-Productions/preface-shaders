// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_hdr_tone_mapping_common.hlsl"
#include "mb_color_correction.hlsl"

struct ps_input_t
{
    float4 m_position : SV_POSITION;
    float2 m_texcoord : TEXCOORD0;
};

ConstantBuffer<cb_push_tone_mapping_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float4 l_color = bindless_tex2d_sample_level(g_push_constants.m_hdr_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_input.m_texcoord, 0);
    l_color.rgb = unpack_lighting(l_color.rgb);

#if EYE_ADAPTATION
    // Get exposure from eye adaptation result
    float l_lum = bindless_tex2d_sample_level(g_push_constants.m_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(g_push_constants.m_lum_uvx, 0.5f), 0).r;
    float l_exposure_scale = g_push_constants.m_key_value / l_lum;
#else
    // Fixed exposure
    float l_exposure_scale = ev100_to_exposure(g_push_constants.m_key_value);
#endif

    l_color.rgb *= l_exposure_scale;

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
    l_color.rgb = apply_color_correction(l_color.rgb, g_push_constants.m_color_correction_lut_srv, g_push_constants.m_color_correction_lut_size);

    return l_color;
}
