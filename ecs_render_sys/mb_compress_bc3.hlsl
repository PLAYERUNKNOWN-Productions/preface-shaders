// Copyright:   PlayerUnknown Productions BV

#include "mb_compress_common.hlsl"
#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

// CBV
ConstantBuffer<cb_compress_bc3_t> g_push_constants  : register(REGISTER_PUSH_CONSTANTS);

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
    RWTexture2D<uint4> l_texure_dst = ResourceDescriptorHeap[g_push_constants.m_texture_dst_uav];

    // Load 4x4 pixel block
    float block_a[16];
    float3 block_rgb[16];
    LoadTexelsRGBA(l_texture_src,
                   (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                   id,
                   g_push_constants.m_input_is_srgb,
                   block_rgb,
                   block_a);

    // Find min/max
    float3 min_color;
    float3 max_color;
    GetMinMaxRGB(block_rgb, min_color, max_color);

    float min_alpha;
    float max_alpha;
    GetMinMaxChannel(block_a, min_alpha, max_alpha);

    // Shift colors to reduce RMS error(do not do it for alpha)
    InsetMinMaxRGB(min_color, max_color);

    // Pack our colors into uints
    uint min_color_565 = ColorTo565(min_color);
    uint max_color_565 = ColorTo565(max_color);
    uint min_alpha_packed = round(min_alpha * 255.0f);
    uint max_alpha_packed = round(max_alpha * 255.0f);

    // Get indices
    uint indices = 0;
    if (min_color_565 < max_color_565)
    {
        indices = GetIndicesRGB(block_rgb, min_color, max_color);
    }

    uint2 output_alpha = uint2((min_alpha_packed << 8) | max_alpha_packed, 0);
    if (min_alpha_packed < max_alpha_packed)
    {
        GetIndicesAlpha(block_a, min_alpha, max_alpha, output_alpha);
    }

    l_texure_dst[id] = uint4(output_alpha.x,
                             output_alpha.y,
                             (min_color_565 << 16) | max_color_565,
                             indices);
}
