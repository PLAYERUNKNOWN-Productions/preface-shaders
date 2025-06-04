// Copyright:   PlayerUnknown Productions BV

#ifndef MBSHADER_HDR_TONE_MAPPING_COMMON_H
#define MBSHADER_HDR_TONE_MAPPING_COMMON_H

#include "../helper_shaders/mb_common.hlsl"

float3 tonemap_reinhard(float3 p_color)
{
    return p_color / (p_color + 1.0f);
}

float3 tonemap_reinhard_jodie(float3 p_color)
{
    float l_luminance = get_luminance(p_color);
    float3 l_tcolor = p_color / (1.0f + p_color);
    return lerp(p_color / (1.0f + l_luminance), l_tcolor, l_tcolor);
}

float3 tonemap_exp(float3 p_color)
{
    return 1.0f - exp2(-p_color);
}

float3 tonemap_uncharted2_partial(float3 p_x)
{
    float l_a = 0.15f;
    float l_b = 0.50f;
    float l_c = 0.10f;
    float l_d = 0.20f;
    float l_e = 0.02f;
    float l_f = 0.30f;

    return ((p_x * (l_a * p_x + l_c * l_b) + l_d * l_e) / (p_x * (l_a * p_x + l_b) + l_d * l_f)) - l_e / l_f;
}

float3 tonemap_uncharted2_filmic(float3 p_color)
{
    float l_exposure_bias = 2.0f;
    float3 l_curr = tonemap_uncharted2_partial(p_color * l_exposure_bias);

    float3 l_w = 11.2f;
    float3 l_white_scale = 1.0f / tonemap_uncharted2_partial(l_w);
    return l_curr * l_white_scale;
}

// From https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
float3 tonemap_aces_approx(float3 p_color)
{
    p_color *= 0.6f;

	float l_a = 2.51f;
    float l_b = 0.03f;
    float l_c = 2.43f;
    float l_d = 0.59f;
    float l_e = 0.14f;
    return saturate((p_color * (l_a * p_color + l_b))/(p_color * (l_c * p_color + l_d) + l_e));
}

// From https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
//
// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const float3x3 g_aces_input_mat =
{
    {0.59719f, 0.35458f, 0.04823f},
    {0.07600f, 0.90834f, 0.01566f},
    {0.02840f, 0.13383f, 0.83777f}
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const float3x3 g_aces_output_mat =
{
    { 1.60475f, -0.53108f, -0.07367f},
    {-0.10208f,  1.10813f, -0.00605f},
    {-0.00327f, -0.07276f,  1.07602f}
};

float3 rrt_and_odt_fit(float3 p_v)
{
    float3 l_a = p_v * (p_v + 0.0245786f) - 0.000090537f;
    float3 l_b = p_v * (0.983729f * p_v + 0.4329510f) + 0.238081f;
    return l_a / l_b;
}

float3 tonemap_aces_fitted(float3 p_color)
{
    p_color = mul(g_aces_input_mat, p_color);

    // Apply RRT and ODT
    p_color = rrt_and_odt_fit(p_color);

    p_color = mul(g_aces_output_mat, p_color);

    // Clamp to [0, 1]
    return saturate(p_color);
}

#endif // MBSHADER_HDR_TONE_MAPPING_COMMON_H
