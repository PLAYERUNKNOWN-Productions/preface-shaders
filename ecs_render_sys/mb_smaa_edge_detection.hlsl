// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//this must be declared before mb_smaa_common.hlsl
ConstantBuffer<cb_push_smaa_edge_detection_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#include "mb_smaa_common.hlsl"
#include "mb_smaa_impl.hlsl"

struct edge_detection_ps_input_t
{
    float4 m_position  : SV_POSITION;
    float2 m_texcoord  : TEXCOORD0;
    float4 m_offset[3] : TEXCOORD1;
};

edge_detection_ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    edge_detection_ps_input_t l_result = (edge_detection_ps_input_t)0;

    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);
    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);

    SMAAEdgeDetectionVS(l_result.m_texcoord, l_result.m_offset);

    return l_result;
}

float2 ps_main(edge_detection_ps_input_t p_input) : SV_TARGET
{
    //do LUMA detection for now
    return SMAALumaEdgeDetectionPS(
        p_input.m_texcoord,
        p_input.m_offset,
        ResourceDescriptorHeap[g_push_constants.m_color_texture_srv],
        ResourceDescriptorHeap[g_push_constants.m_depth_texture_srv]);
}
