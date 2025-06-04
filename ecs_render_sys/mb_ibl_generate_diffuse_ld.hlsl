// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position_cs        : SV_POSITION;
    float3 m_position_ws_local  : POSITION;
};

ConstantBuffer<cb_push_generate_diffuse_ld_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    // Get camera cb
    ConstantBuffer<cb_camera_light_probe_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    l_result.m_position_cs = get_fullscreen_triangle_position(p_vertex_id);
    float4 l_position_ws_local = mul(l_result.m_position_cs, l_camera.m_inv_view_proj_local);
    l_result.m_position_ws_local = l_position_ws_local.xyz;

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_view_dir = normalize(p_input.m_position_ws_local);

    float3 l_n = l_view_dir;

    float4 l_val = integrate_diffuse_ld(g_push_constants.m_texture_srv,
                                        l_n,
                                        g_push_constants.m_num_samples,
                                        g_push_constants.m_mip_level);

    return l_val;
}
