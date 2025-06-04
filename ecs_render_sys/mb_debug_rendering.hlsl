// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//this must be declared before mb_smaa_common.hlsl
ConstantBuffer<cb_push_debug_rendering_t>       g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

struct blend_ps_input_t
{
    float4 m_sv_position    : SV_POSITION;
    float4 m_color          : COLOR;
};

blend_ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    blend_ps_input_t l_result = (blend_ps_input_t)0;

    //fetch vertex from vertex buffer
    StructuredBuffer<sb_debug_rendering_vertex_t> l_vertices = ResourceDescriptorHeap[g_push_constants.m_vertex_buffer_srv];
    sb_debug_rendering_vertex_t l_vertex = l_vertices[p_vertex_id];

    //transform to projection space
    l_result.m_sv_position = mul(l_vertex.m_position, g_push_constants.m_view_proj_matrix);

    l_result.m_color = l_vertex.m_color;

    return l_result;
}

float4 ps_main(blend_ps_input_t p_input) : SV_TARGET
{
    return float4(p_input.m_color.x, p_input.m_color.y, p_input.m_color.z, 1.f);
}
