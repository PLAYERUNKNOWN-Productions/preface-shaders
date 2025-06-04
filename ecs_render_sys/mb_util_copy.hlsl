// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"

struct ps_input_t
{
    float4 m_position   : SV_POSITION;
    float2 m_texcoord   : TEXCOORD0;
};

ConstantBuffer<cb_push_copy_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_main(uint p_vertex_id : SV_VertexID)
{
    ps_input_t l_result;

    l_result.m_texcoord = get_fullscreen_triangle_texcoord(p_vertex_id);
    l_result.m_position = get_fullscreen_triangle_position(p_vertex_id);

    return l_result;
}

float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    ConstantBuffer<cb_test_values_t> l_test_values = ResourceDescriptorHeap[CBV_TEST_VALUES];
    const int l_kernel_size = l_test_values.m_int_val.y;
    const int l_pixel_step = l_test_values.m_int_val.z;

    // Copy with bilinear filter
    float4 l_color = 0;
    float l_weight = 0;
    for(int l_x = -l_kernel_size; l_x <= l_kernel_size; ++l_x)
    {
        for(int l_y = -l_kernel_size; l_y <= l_kernel_size; ++l_y)
        {
            float2 l_offset = (float)l_pixel_step * float2(l_x, l_y) / 4096.0f;

            float l_uv_scale = 1.0;
            float2 l_uv = l_uv_scale * (p_input.m_texcoord + l_offset);

            // Sample
            l_color += bindless_tex2d_sample(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv);

            l_weight += 1;
        }
    }

    return l_color / l_weight;
}
