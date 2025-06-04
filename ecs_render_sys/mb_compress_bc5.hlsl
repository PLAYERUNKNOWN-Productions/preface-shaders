// Copyright:   PlayerUnknown Productions BV

#include "mb_compress_common.hlsl"
#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

// CBV
ConstantBuffer<cb_compress_bc5_t> g_push_constants  : register(REGISTER_PUSH_CONSTANTS);

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
    float block_r[16];
    float block_g[16];
    LoadTexelsRG(l_texture_src,
                 (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                 1.0f / float2(g_push_constants.m_src_resolution_x, g_push_constants.m_src_resolution_y),
                 id,
                 block_r,
                 block_g);

    // Find min/max
    float min_r;
    float max_r;
    float min_g;
    float max_g;
    GetMinMaxRG(block_r, block_g, min_r, max_r, min_g, max_g);

    // Pack our colors into uints
    uint min_r_packed = round(min_r * 255.0f);
    uint max_r_packed = round(max_r * 255.0f);
    uint min_g_packed = round(min_g * 255.0f);
    uint max_g_packed = round(max_g * 255.0f);

    uint2 output_r = uint2((min_r_packed << 8) | max_r_packed, 0);
    uint2 output_g = uint2((min_g_packed << 8) | max_g_packed, 0);

    // Get indices
    if (min_r_packed < max_r_packed)
    {
        GetIndicesAlpha(block_r, min_r, max_r, output_r);
    }

    if (min_g_packed < max_g_packed)
    {
        GetIndicesAlpha(block_g, min_g, max_g, output_g);
    }

    l_texure_dst[id] = uint4(output_r.x, output_r.y, output_g.x, output_g.y);
}
