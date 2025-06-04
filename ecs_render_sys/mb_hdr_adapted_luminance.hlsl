// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position       : SV_POSITION;
};

ConstantBuffer<cb_push_adapted_luminance_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

float ps_main(ps_input_t p_input) : SV_TARGET
{
    // Get the log average luminance of the current frame
    float l_cur_lum = bindless_tex2d_sample_level(g_push_constants.m_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(0.5f, 0.5f)).x;
#if LOG_LUMINANCE
    l_cur_lum = exp(l_cur_lum);
#endif

    // Clamp to limit the range and avoid NaN
    l_cur_lum = clamp(l_cur_lum, g_push_constants.m_min_avg_lum, g_push_constants.m_max_avg_lum);

    // Temporal luminance adaptation
    float l_prev_adapted_lum = bindless_tex2d_sample_level(g_push_constants.m_adapted_lum_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(g_push_constants.m_prev_adapted_lum_uvx, 0.5f)).x;
    float l_adapted_lum = compute_luminance_adaptation(l_prev_adapted_lum, l_cur_lum, g_push_constants.m_speed_up, g_push_constants.m_speed_down, g_push_constants.m_delta_time);
    return l_adapted_lum;
}
