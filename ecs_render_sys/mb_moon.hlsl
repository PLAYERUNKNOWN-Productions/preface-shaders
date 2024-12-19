// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
    float4 m_position               : SV_POSITION;
    float2 m_texcoord               : TEXCOORD0;
    float3 m_position_camera_local  : POSITION;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_moon_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
float4 get_billboard_position(uint p_vertex_id,
                              float3 p_billboard_pos,
                              float p_billboard_size,
                              float3 p_camera_pos,
                              float4x4 p_view_local,
                              float4x4 p_proj,
                              float2 p_resolution)
{
    static const float2 g_quad_vertices[6] =
    {
        float2(-0.5f,  0.5f),
        float2( 0.5f,  0.5f),
        float2(-0.5f, -0.5f),
        float2( 0.5f,  0.5f),
        float2( 0.5f, -0.5f),
        float2(-0.5f, -0.5f)
    };

    float4 l_pos_ps = mul(float4(p_billboard_pos - p_camera_pos, 1.0), p_view_local);
    float4 l_pos_cs = mul(l_pos_ps, p_proj);
    float3 l_pos_ndc = l_pos_cs.xyz / l_pos_cs.w;

    float2 l_offset = g_quad_vertices[p_vertex_id] * p_billboard_size;
    float2 l_final_pos = l_pos_ndc.xy + l_offset * float2(2.0f / p_resolution.x, 2.0f / p_resolution.y);

    return float4(l_final_pos, l_pos_ndc.z, 1.0f);
}

//-----------------------------------------------------------------------------
float2 get_billboard_texcoord(uint p_vertex_id)
{
    static const float2 g_quad_texcoords[6] = 
    {
        float2(0.0f, 0.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 1.0f),
        float2(1.0f, 0.0f),
        float2(1.0f, 1.0f),
        float2(0.0f, 1.0f)
    };

    return g_quad_texcoords[p_vertex_id];
}

//-----------------------------------------------------------------------------
// [Daniel et al. 2012, "Single-Pass Rendering of Day and Night Sky Phenomena"]
float retrodirective_function(float p_phi, float p_g)
{
    if (p_phi < M_PI * 0.5f)
    {
        float l_tan_phi = tan(p_phi);
        float l_exp_term = exp(-p_g / l_tan_phi);
        return (2.0f - (l_tan_phi * (2.0f / p_g)) * (1.0f - l_exp_term) * (3.0f - l_exp_term));
    }
    else
    {
        return 1.0f;
    }
}

//-----------------------------------------------------------------------------
// [Daniel et al. 2012, "Single-Pass Rendering of Day and Night Sky Phenomena"]
float scattering_function(float p_phi, float p_t)
{
    float l_abs_phi = abs(p_phi);
    float l_sin_phi = sin(l_abs_phi);
    float l_cos_phi = cos(l_abs_phi);

    float l_cosine_adjustment = 1.0f - 0.5f * l_cos_phi;
    return (l_sin_phi + (M_PI - l_abs_phi) * l_cos_phi) / M_PI + p_t * l_cosine_adjustment * l_cosine_adjustment;
}

//-----------------------------------------------------------------------------
// [Daniel et al. 2012, "Single-Pass Rendering of Day and Night Sky Phenomena"]
float hapke_lommel_seeliger_brdf(float p_cos_theta_i, float p_cos_theta_r, float p_phi)
{
    const float c_g = 0.6f;
    float l_b = retrodirective_function(p_phi, c_g);

    const float c_t = 0.1f;
    float l_s = scattering_function(p_phi, c_t);

    float l_brdf = 2.0f / (3.0f * M_PI) * l_b * l_s / (1.0f + p_cos_theta_r / p_cos_theta_i);
    return l_brdf;
}

//-----------------------------------------------------------------------------
// [Daniel et al. 2012, "Single-Pass Rendering of Day and Night Sky Phenomena"]
float earthshine_intensity_function(float p_phi)
{
    float l_phi_2 = p_phi * p_phi;
    float l_phi_3 = l_phi_2 * p_phi;
    return -0.0061f * l_phi_3 + 0.0289f * l_phi_2 - 0.0105f * sin(p_phi);
}

//-----------------------------------------------------------------------------
// [Daniel et al. 2012, "Single-Pass Rendering of Day and Night Sky Phenomena"]
float3 compute_moon_lighting(float3 p_k_lambda, float3 p_albedo, float p_cos_theta_i, float p_cos_theta_r, float p_phi)
{
    // Hapke-Lommel-Seeliger BRDF
    float l_brdf = hapke_lommel_seeliger_brdf(p_cos_theta_i, p_cos_theta_r, p_phi);

    // Earthshine
    const float3 l_beta_e = float3(0.88f, 0.96f, 1.0f);
    float3 l_earthshine = earthshine_intensity_function(p_phi) * l_beta_e;

    return p_k_lambda * p_albedo * l_brdf  + l_earthshine;
}

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    l_result.m_position = get_billboard_position(p_vertex_id, g_push_constants.m_moon_position,
                                                 g_push_constants.m_billboard_size,
                                                 l_camera.m_camera_pos,
                                                 l_camera.m_view_local,
                                                 g_push_constants.m_proj,
                                                 float2(l_camera.m_resolution_x, l_camera.m_resolution_y));
    l_result.m_texcoord = get_billboard_texcoord(p_vertex_id);
    l_result.m_position_camera_local = g_push_constants.m_moon_position - l_camera.m_camera_pos;

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float4 l_albedo = bindless_tex2d_sample_level(g_push_constants.m_albedo_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord);

    // Smooth edge transition
    float l_distance_from_center = length(p_input.m_texcoord - float2(0.5f, 0.5f));
    float l_smooth_edge_transition = smoothstep(g_push_constants.m_edge_transition_radius, g_push_constants.m_edge_transition_radius - g_push_constants.m_edge_transition_width, l_distance_from_center);
    float l_alpha = l_smooth_edge_transition * l_albedo.a;

    // Only write depth for the Moon area
    if (l_alpha < 0.01f)
    {
        discard;
    }

    float3 l_normal_ts = 0;
    l_normal_ts.xy = bindless_tex2d_sample_level(g_push_constants.m_normal_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_input.m_texcoord).xy;
    l_normal_ts.xy = l_normal_ts.xy * 2.0f - 1.0f;
    l_normal_ts.z = sqrt(1.0f - saturate(dot(l_normal_ts.xy, l_normal_ts.xy)));

    // Lighting
    float l_cos_theta_i = saturate(dot(l_normal_ts, g_push_constants.m_light_dir));
    float3 l_view_dir = normalize(-p_input.m_position_camera_local);
    float l_cos_theta_r = saturate(dot(l_normal_ts, l_view_dir));
    float3 l_final_lighting = compute_moon_lighting(g_push_constants.m_k_lambda, l_albedo.rgb, l_cos_theta_i, l_cos_theta_r, g_push_constants.m_lunar_phase_angle);

    l_final_lighting = pack_lighting(l_final_lighting);
    return float4(l_final_lighting, l_alpha);
}
