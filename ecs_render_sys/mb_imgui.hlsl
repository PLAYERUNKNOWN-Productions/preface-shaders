// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------
struct vs_input_t
{
    float2 m_position : POSITION;
    float4 m_color    : COLOR0;
    float2 m_uv       : TEXCOORD0;
};

struct ps_input_t
{
    float4 m_position : SV_POSITION;
    float4 m_color    : COLOR0;
    float2 m_uv       : TEXCOORD0;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_imgui_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Vertex shader
//-----------------------------------------------------------------------------
ps_input_t vs_main(vs_input_t input)
{
    ps_input_t result;
    result.m_position = mul(g_push_constants.m_projection, float4(input.m_position, 0.f, 1.f));
    result.m_color    = input.m_color;
    result.m_uv       = input.m_uv;
    return result;
}

//-----------------------------------------------------------------------------
// Pixel shader
//-----------------------------------------------------------------------------
float4 ps_main(ps_input_t input) : SV_TARGET
{
    SamplerState sampler = SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP];
    float4 texture_sample = bindless_tex2d_sample(g_push_constants.m_texture_srv, sampler, input.m_uv);
    return input.m_color * texture_sample;
}
