// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//this must be declared before mb_smaa_common.hlsl
ConstantBuffer<cb_push_smaa_edge_detection_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#include "mb_smaa_common.hlsl"
#include "mb_smaa_impl.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct edge_detection_ps_input_t
{
    float4 m_svPosition : SV_POSITION;
    float2 m_texcoord   : TEXCOORD0;
    float4 m_offset[3]  : TEXCOORD1;
};

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------

edge_detection_ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    edge_detection_ps_input_t l_result = (edge_detection_ps_input_t)0;

    float2 texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    float4 position = get_fullscreen_triangle_position(p_vertex_id);

    l_result.m_texcoord = texcoord;
    SMAAEdgeDetectionVS(position, texcoord, l_result.m_svPosition, l_result.m_offset);

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

float4 ps_main(edge_detection_ps_input_t p_input) : SV_TARGET
{
    //do LUMA detection for now
    uint l_color_texture_index = g_push_constants.m_color_texture_srv;
    uint l_depth_texture_index = g_push_constants.m_depth_texture_srv;

    return SMAALumaEdgeDetectionPS(
        p_input.m_texcoord, 
        p_input.m_offset, 
        ResourceDescriptorHeap[l_color_texture_index],
        ResourceDescriptorHeap[l_depth_texture_index]);
}

