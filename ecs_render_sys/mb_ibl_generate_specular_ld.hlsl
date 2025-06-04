// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position_cs        : SV_POSITION;
    float3 m_position_ws_local  : POSITION;
};

ConstantBuffer<cb_push_generate_specular_ld_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

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

    float4 l_val = 0;
    if (g_push_constants.m_level == 0) // For mip 0, just do copy
    {
        l_val = bindless_texcube_sample_level(g_push_constants.m_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_view_dir, 0);
    }
    else
    {
        float3 l_n = l_view_dir;

        // Remove view-dependency from GGX, effectively making the BSDF isotropic
        float3 l_v = l_n;

        // Get linear roughness by current mip level
        float l_linear_roughness = mipmap_level_to_linear_roughness(g_push_constants.m_level, g_push_constants.m_mip_count);

        // Clamp and remap roughness
        float l_alpha = clamp_and_remap_roughness(l_linear_roughness);

        // Get sample count by current mip level
        uint l_sample_count = get_ibl_runtime_filter_sample_count(g_push_constants.m_level);

        l_val = integrate_specular_ld(g_push_constants.m_texture_srv, l_n, l_v, l_alpha,
                                      g_push_constants.m_inv_omega_p,
                                      l_sample_count // Must be a Fibonacci number
                                      );
    }

    return l_val;
}
