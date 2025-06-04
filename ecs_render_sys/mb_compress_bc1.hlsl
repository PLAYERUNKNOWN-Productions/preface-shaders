// Copyright:   PlayerUnknown Productions BV

#include "mb_compress_common.hlsl"
#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

// CBV
ConstantBuffer<cb_compress_bc1_t> g_push_constants  : register(REGISTER_PUSH_CONSTANTS);

[numthreads(COMPRESS_THREAD_GROUP_SIZE, COMPRESS_THREAD_GROUP_SIZE, 1)]
void cs_main(uint2 id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (id.x >= g_push_constants.m_dst_resolution_x ||
        id.y >= g_push_constants.m_dst_resolution_y)
    {
        return;
    }

    // Get resources
    Texture2D l_texture_src = ResourceDescriptorHeap[g_push_constants.m_texture_src_srv];
    RWTexture2D<uint2> l_texure_dst = ResourceDescriptorHeap[g_push_constants.m_texture_dst_uav];

    // Load 4x4 pixel block
    float3 colors[16];
    LoadTexelsRGB(l_texture_src,
                  (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                  1.0f / float2(g_push_constants.m_src_resolution_x, g_push_constants.m_src_resolution_y),
                  id,
                  g_push_constants.m_input_is_srgb,
                  colors);

    // Find min/max
    float3 min_color;
    float3 max_color;
    GetMinMaxRGB(colors, min_color, max_color);

    // Shift colors to reduce RMS error
    InsetMinMaxRGB(min_color, max_color);

    // Pack our colors into uints
    uint min_color_565 = ColorTo565(min_color);
    uint max_color_565 = ColorTo565(max_color);

    // Get indices
    uint indices = 0;
    if (min_color_565 < max_color_565)
    {
        indices = GetIndicesRGB(colors, min_color, max_color);
    }

    l_texure_dst[id] = uint2((min_color_565 << 16) | max_color_565, indices);
}
