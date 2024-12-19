// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
    float4 m_position       : SV_POSITION;
    float2 m_texcoord       : TEXCOORD0;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_plot_adapted_luminance_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

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

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    // Get the relative index of current pixel in a MB_MAX_ADAPTED_LUM_FRAMES pixel wide texture
    uint l_pixel_index = p_input.m_texcoord.x * MB_MAX_ADAPTED_LUM_FRAMES;

    // Plot current luminance at the end
    uint l_frame_index = (l_pixel_index + (g_push_constants.m_cur_adapted_lum_index + 1)) % MB_MAX_ADAPTED_LUM_FRAMES;

    // Find the texture to fetch the luminance
    uint l_texture_index = ((l_frame_index % 2) == 0) ? g_push_constants.m_adapted_lum_texture_srv.x : g_push_constants.m_adapted_lum_texture_srv.y;

    // texture0.width + texture1.width = MB_MAX_ADAPTED_LUM_FRAMES
    const uint l_texture_width = MB_MAX_ADAPTED_LUM_FRAMES >> 1;

    // Fetch luminance value
    float l_lum_uvx = (0.5f + l_frame_index / 2) / l_texture_width;
    float l_lum = bindless_tex2d_sample_level(l_texture_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(l_lum_uvx, 0.5f)).x;

    // Get the target y position to plot based on the luminance value
    float l_local_uv_y = 1.0f - saturate((l_lum - g_push_constants.m_lum_bounds.x) / (g_push_constants.m_lum_bounds.y - g_push_constants.m_lum_bounds.x));
    uint l_target_local_pos_y = l_local_uv_y * g_push_constants.m_dst_resolution_y;

    // Plot the target pixel
    uint l_local_pos_y = g_push_constants.m_dst_resolution_y * p_input.m_texcoord.y;
    if (l_local_pos_y == l_target_local_pos_y)
    {
        return float4(0, 1, 0, 1);
    }
    else
    {
        return float4(0, 0, 0, 1);
    }
}
