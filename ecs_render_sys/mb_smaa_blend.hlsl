// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//this must be declared before mb_smaa_common.hlsl
ConstantBuffer<cb_push_smaa_blend_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#include "mb_smaa_common.hlsl"
#include "mb_smaa_impl.hlsl"

struct blend_ps_input_t
{
    float4 m_position : SV_POSITION;
    float2 m_texcoord : TEXCOORD0;
    float4 m_offset   : TEXCOORD1;
};

blend_ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    blend_ps_input_t l_result = (blend_ps_input_t)0;

    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);
    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);

    SMAANeighborhoodBlendingVS(l_result.m_texcoord, l_result.m_offset);

    return l_result;
}

float4 ps_main(blend_ps_input_t p_input) : SV_TARGET
{
    return SMAANeighborhoodBlendingPS(
        p_input.m_texcoord,
        p_input.m_offset,
        ResourceDescriptorHeap[g_push_constants.m_color_texture_srv],
        ResourceDescriptorHeap[g_push_constants.m_blend_texture_srv]);
}
