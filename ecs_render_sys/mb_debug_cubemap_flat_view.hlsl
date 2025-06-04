// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position       : SV_POSITION;
    float3 m_lookup_vector  : TEXCOORD0;
};

ConstantBuffer<cb_push_debug_cubemap_flat_view_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_position = get_fullscreen_quad_position(p_vertex_id);
    l_result.m_lookup_vector = get_cube_lookup_vector(g_push_constants.m_face_id, p_vertex_id);

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float4 l_color = bindless_texcube_sample_level(g_push_constants.m_cubemap_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_lookup_vector, 0);
    return l_color;
}
