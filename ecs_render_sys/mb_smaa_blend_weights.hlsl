// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

//this must be declared before mb_smaa_common.hlsl
ConstantBuffer<cb_push_smaa_blend_weights_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#include "mb_smaa_common.hlsl"
#include "mb_smaa_impl.hlsl"

struct blend_weight_ps_input_t
{
    float4 m_position  : SV_POSITION;
    float2 m_texcoord  : TEXCOORD0;
    float2 m_pixcoord  : TEXCOORD1;
    float4 m_offset[3] : TEXCOORD2;
};

blend_weight_ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    blend_weight_ps_input_t l_result = (blend_weight_ps_input_t)0;

    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);
    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);

    SMAABlendingWeightCalculationVS(l_result.m_texcoord, l_result.m_pixcoord, l_result.m_offset);

    return l_result;
}

float4 ps_main(blend_weight_ps_input_t p_input) : SV_TARGET
{
    return SMAABlendingWeightCalculationPS(
        p_input.m_texcoord,
        p_input.m_pixcoord,
        p_input.m_offset,
        ResourceDescriptorHeap[g_push_constants.m_edges_texture_srv],
        ResourceDescriptorHeap[g_push_constants.m_area_texture_srv],
        ResourceDescriptorHeap[g_push_constants.m_search_texture_srv],
        int4(0, 0, 0, 0));//no subsample indices needed so far
}
