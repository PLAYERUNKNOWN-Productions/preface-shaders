// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MB_SHADER_POSTPROCESS_VS_HLSL
#define MB_SHADER_POSTPROCESS_VS_HLSL

#include "mb_lighting_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------
struct ps_input_t
{
    float4 m_position : SV_POSITION;
    float2 m_texcoord : TEXCOORD0;
};

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

#endif