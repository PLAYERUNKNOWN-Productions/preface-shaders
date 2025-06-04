// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position       : SV_POSITION;
    float2 m_texcoord       : TEXCOORD0;
};

ConstantBuffer<cb_push_hdr_off_t>   g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float4 l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_input.m_texcoord);

    // Unpack lighting
    l_color.rgb = unpack_lighting(l_color.rgb);

    // Gamma correction
    l_color.rgb = linear_to_gamma(l_color.rgb);

    return l_color;
}
