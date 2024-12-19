// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_hdr_tone_mapping_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
    float4 m_position   : SV_POSITION;
    float2 m_texcoord   : TEXCOORD0;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_debug_visualize_texture_t>   g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

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

#define COLOR_TABLE_SIZE 32

static float3 s_color_table[COLOR_TABLE_SIZE] =
{
    float3(0.0f, 0.0f, 0.0f),   // black
    float3(0.0f, 1.0f, 0.0f),   // Green
    float3(0.0f, 0.0f, 1.0f),   // Blue
    float3(1.0f, 1.0f, 0.0f),   // Yellow
    float3(1.0f, 0.0f, 1.0f),   // Magenta
    float3(0.0f, 1.0f, 1.0f),   // Cyan
    float3(1.0f, 0.5f, 0.0f),   // Orange
    float3(0.5f, 0.0f, 1.0f),   // Purple
    float3(0.0f, 0.5f, 0.5f),   // Teal
    float3(0.4f, 0.4f, 0.0f),   // Gray
    float3(0.0f, 0.4f, 0.4f),   // Light Gray
    float3(0.4f, 0.0f, 0.4f),   // Dark Gray
    float3(0.5f, 0.0f, 0.0f),   // Maroon
    float3(0.0f, 0.5f, 0.0f),   // Olive
    float3(0.0f, 0.0f, 0.5f),   // Navy
    float3(0.0f, 0.5f, 0.5f),   // Sea Green
    float3(0.5f, 0.0f, 0.5f),   // Indigo
    float3(0.5f, 0.5f, 0.0f),   // Olive Drab
    float3(0.5f, 0.0f, 0.5f),   // Dark Magenta
    float3(0.0f, 0.5f, 0.5f),   // Medium Aquamarine
    float3(0.5f, 0.5f, 0.0f),   // Dark Khaki
    float3(0.2f, 0.5f, 0.2f),   // Forest Green
    float3(0.0f, 0.2f, 0.0f),   // Dark Green
    float3(0.5f, 0.2f, 0.0f),   // Brown
    float3(0.0f, 0.2f, 0.5f),   // Midnight Blue
    float3(0.2f, 0.0f, 0.5f),   // Dark Slate Blue
    float3(0.5f, 0.0f, 0.2f),   // Dark Red
    float3(0.2f, 0.5f, 0.5f),   // Steel Blue
    float3(0.5f, 0.2f, 0.5f),   // Medium Purple
    float3(0.2f, 0.5f, 0.2f),   // Dark Olive Green
    float3(0.2f, 0.7f, 0.0f),   // Black (Excluded)
    float3(0.7f, 0.0f, 0.7f),   // White (Excluded)
};

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    float3 l_color = 0;
    if (g_push_constants.m_is_uint_format != 0)
    {
        if (g_push_constants.m_format_num_channels == 1)
        {
            Texture2D<uint> l_texture = ResourceDescriptorHeap[g_push_constants.m_src_texture_srv];
            uint l_classification = l_texture.Load(uint3(p_input.m_texcoord * g_push_constants.m_src_texture_size, 0)); 
            l_color.rgb = s_color_table[l_classification % COLOR_TABLE_SIZE];
        }
        else if (g_push_constants.m_format_num_channels == 2)
        {
            Texture2D<uint2> l_texture = ResourceDescriptorHeap[g_push_constants.m_src_texture_srv];
            uint2 l_instance_primitive_id = l_texture.Load(uint3(p_input.m_texcoord * g_push_constants.m_src_texture_size, 0));
            l_color.rgb = s_color_table[l_instance_primitive_id.g % COLOR_TABLE_SIZE];

#if 0
            uint l_instance_id = 0;
            bool l_front_face = false;
            bool l_wind = false;
            bool l_wind_small = false;
            unpack_instance_id_pixel_options(l_instance_primitive_id.r, l_instance_id, l_front_face, l_wind, l_wind_small);
            
            float3 l_wind_color = l_wind ? s_color_table[1] : (float3)0;
            float3 l_wind_small_color = l_wind_small ? s_color_table[2] : (float3)0;

            l_color.rgb = l_wind_color + l_wind_small_color;
#endif
        }
    }
    else
    {
        if (g_push_constants.m_is_texture_array != 0)
        {
            Texture2DArray l_texture_array = ResourceDescriptorHeap[g_push_constants.m_src_texture_srv];
            l_color = l_texture_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float3(p_input.m_texcoord, g_push_constants.m_texture_array_index), g_push_constants.m_texture_mip_index).rgb;
        }
        else
        {
            l_color = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_input.m_texcoord, g_push_constants.m_texture_mip_index).rgb;
        }

        if (g_push_constants.m_format_num_channels == 1)
        {
            l_color.gb = l_color.r;
        }
    }

#if TONE_BY_RANGE
    if (g_push_constants.m_is_hdr_color == 1)
    {
        float3 l_min = bindless_tex2d_sample_level(g_push_constants.m_lowest_value_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(0.5f, 0.5f)).rgb;
        float3 l_max = bindless_tex2d_sample_level(g_push_constants.m_highest_value_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(0.5f, 0.5f)).rgb;

        const float l_lum = 0.5f * (get_luminance(l_min) + get_luminance(l_max));
        const float l_exposure_key_value = 0.5f;
        const float l_exposure_scale = l_exposure_key_value / l_lum;

        // Apply exposure scale
        l_color.xyz *= l_exposure_scale;

        // Tone mapping
        l_color.xyz = tonemap_exp(l_color.xyz);
    }
    else // Single channel
    {
        if (g_push_constants.m_is_uint_format != 0)
        {
                Texture2D<uint> l_lowest_texture = ResourceDescriptorHeap[g_push_constants.m_lowest_value_texture_srv];
                uint l_min = l_lowest_texture.Load(uint3(0, 0, 0));
                Texture2D<uint> l_highest_texture = ResourceDescriptorHeap[g_push_constants.m_highest_value_texture_srv];
                uint l_max = l_highest_texture.Load(uint3(0, 0, 0));

                float3 l_range = l_max - l_min;
                l_color = clamp(l_color - l_min, 0, l_range) / l_range;
        }
        else
        {
            float3 l_min = bindless_tex2d_sample_level(g_push_constants.m_lowest_value_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(0.5f, 0.5f)).r;
            float3 l_max = bindless_tex2d_sample_level(g_push_constants.m_highest_value_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], float2(0.5f, 0.5f)).r;

            float3 l_range = l_max - l_min;
            l_color = clamp(l_color - l_min, 0, l_range) / l_range;
        }
    }
#endif

    // Swtich channels on/off
    l_color *= g_push_constants.m_channel_swtiches;

    return float4(l_color, 1);
}
