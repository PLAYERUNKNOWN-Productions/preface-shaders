// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position : SV_POSITION;
    float2 m_texcoord : TEXCOORD0;
};

ConstantBuffer<cb_push_debug_luminance_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

// [0, 1] -> [blue, red]
float3 get_temperatured_color_by_range(float p_value)
{
    float3 l_hsl = float3((1.0f - p_value) * 0.7f, 1, 1);
    return hsl_to_rgb(l_hsl);
}

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).rgb;
    l_color.rgb = unpack_lighting(l_color.rgb);

    // Get luminance from hdr color
    float l_lum = get_luminance(l_color);

    // Get color temperature like outputting color
    l_lum = clamp(l_lum, g_push_constants.m_lum_bounds.x, g_push_constants.m_lum_bounds.y);
    float l_val = (l_lum - g_push_constants.m_lum_bounds.x) / (g_push_constants.m_lum_bounds.y - g_push_constants.m_lum_bounds.x);

    return float4(get_temperatured_color_by_range(l_val), 1.0f);
}
