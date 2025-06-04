// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position_cs        : SV_POSITION;
    float3 m_position_ws_local  : POSITION;
};

ConstantBuffer<cb_push_background_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    l_result.m_position_cs = get_fullscreen_triangle_position(p_vertex_id);
    float4 l_position_ws_local = mul(l_result.m_position_cs, l_camera.m_inv_view_proj_local);
    l_position_ws_local = mul(l_position_ws_local, l_camera.m_align_ground_rotation);
    l_result.m_position_ws_local = l_position_ws_local.xyz;

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_view_dir = normalize(p_input.m_position_ws_local);

    float4 l_color = bindless_texcube_sample_level(g_push_constants.m_cubemap_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_view_dir, 0);

    // Apply intensity multiplier
    float l_multiplier = ev100_to_luminance(g_push_constants.m_exposure_value);
    l_color.rgb *= l_multiplier;

    l_color.rgb = pack_lighting(l_color.rgb);
    return l_color;
}
