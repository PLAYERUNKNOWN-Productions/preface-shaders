// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_hdr_tone_mapping_common.hlsl"
#include "mb_postprocess_vs.hlsl"
#include "mb_color_correction.hlsl"

ConstantBuffer<cb_push_exposure_fusion_final_combine_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 sample_blend(float2 p_texcoord)
{
    return bindless_tex2d_sample_level(g_push_constants.m_blend_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_texcoord, 0).rgb;
}

float3 sample_exposures(float2 p_texcoord)
{
    return bindless_tex2d_sample_level(g_push_constants.m_exposures_mip_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_texcoord, 0).rgb;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
#if MB_EXPOSURE_FUSION_USE_GUIDED_FILTERING
    // Guided upsampling.
    // See https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/
    float l_momentX = 0.0;
    float l_momentY = 0.0;
    float l_momentX2 = 0.0;
    float l_momentXY = 0.0;
    float l_ws = 0.0;
    for (int l_dy = -1; l_dy <= 1; l_dy++) {
        for (int l_dx = -1; l_dx <= 1; l_dx++) {
            float l_x = sample_exposures(p_input.m_texcoord + float2(l_dx, l_dy) * g_push_constants.m_inv_pixel_size).y;
            float l_y = sample_blend    (p_input.m_texcoord + float2(l_dx, l_dy) * g_push_constants.m_inv_pixel_size).x;
            float l_w = exp(-0.5 * (l_dx*l_dx + l_dy*l_dy) / (0.7*0.7));
            l_momentX += l_x * l_w;
            l_momentY += l_y * l_w;
            l_momentX2 += l_x * l_x * l_w;
            l_momentXY += l_x * l_y * l_w;
            l_ws += l_w;
        }
    }
    l_momentX  /= l_ws;
    l_momentY  /= l_ws;
    l_momentX2 /= l_ws;
    l_momentXY /= l_ws;
    float l_A = (l_momentXY - l_momentX * l_momentY) / (max(l_momentX2 - l_momentX * l_momentX, 0.0) + 0.00001);
    float l_B = l_momentY - l_A * l_momentX;
#endif

    // Get exposure from eye adaptation result
    float l_lum = bindless_tex2d_sample_level(g_push_constants.m_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(g_push_constants.m_lum_uvx, 0.5f), 0).r;
    float l_exposure_scale = g_push_constants.m_key_value / l_lum;

    float3 l_color = bindless_tex2d_sample_level(g_push_constants.m_hdr_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord, 0).rgb;
    l_color = unpack_lighting(l_color);
    float3 l_exposed_color = l_color * l_exposure_scale;

    // Apply local exposure adjustment as a crude multiplier on all RGB channels.
    // This is... generally pretty wrong, but enough for the demo purpose.
    float3 l_texel_original = linear_to_gamma(tonemap_aces_fitted(l_exposed_color));
    float l_luminance = get_luminance(l_texel_original) + 0.0001;

#if MB_EXPOSURE_FUSION_USE_GUIDED_FILTERING
    float l_final_multiplier = max(l_A * l_luminance + l_B, 0.0) / l_luminance;
#else
    float l_blend = sample_blend(p_input.m_texcoord).r;
    float l_final_multiplier = max(l_blend, 0) / l_luminance;
#endif

    // This is a hack to prevent super dark pixels getting boosted by a lot and showing compression artifacts.
    float l_lerp_to_unity_threshold = 0.007;
    l_final_multiplier = l_luminance > l_lerp_to_unity_threshold ? l_final_multiplier :
        lerp(1.0, l_final_multiplier, (l_luminance / l_lerp_to_unity_threshold) * (l_luminance / l_lerp_to_unity_threshold));

    float3 l_texel_final = linear_to_gamma(tonemap_aces_fitted(l_exposed_color * l_final_multiplier));

    l_texel_final = apply_color_correction(l_texel_final, g_push_constants.m_color_correction_lut_srv, g_push_constants.m_color_correction_lut_size);

    return float4(l_texel_final, 1.0);
}
