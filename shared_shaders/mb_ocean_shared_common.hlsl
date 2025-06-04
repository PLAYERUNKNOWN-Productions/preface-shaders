// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_OCEAN_SHARED_COMMON
#define MB_SHADER_OCEAN_SHARED_COMMON

#include "mb_shared_buffers.hlsl"

// This is HLSL shared with C implementation
// Use of only shared functionality is permitted!

// Define math types
#ifdef __cplusplus

#include "mb_shared_types_define.hlsl"
#define sb_quadrilateral_patch const mb_shared_buffers::sb_quadrilateral_patch&

namespace mb_ocean_shared_common
{
#endif

float2 remapTilePosition(float2 tile_position, uint weight_mask)
{
    float2 ret = (float2)0;

    uint2 mask = uint2(weight_mask >> 16, weight_mask & 0xFFFF);

    const uint num_elements = 12;

    float elements[num_elements] =
    {
        tile_position.x,                        // x
        1.0f - tile_position.x,                 // inv_x
        tile_position.x * 0.5f,                 // half_x
        0.5f + tile_position.x * 0.5f,          // shifted_half_x
        1.0f - tile_position.x * 0.5f,          // inv_half_x
        1.0f - (0.5f + tile_position.x * 0.5f), // inv_shifted_half_x

        tile_position.y,                        // y
        1.0f - tile_position.y,                 // inv_y
        tile_position.y * 0.5f,                 // half_y
        0.5f + tile_position.y * 0.5f,          // shifted_half_y
        1.0f - tile_position.y * 0.5f,          // inv_half_y
        1.0f - (0.5f + tile_position.y * 0.5f), // inv_shifted_half_y
    };

    for (uint i = 0; i < num_elements; ++i)
    {
        if (mask.x & (1u << i))
            ret.x += elements[i];

        if (mask.y & (1u << i))
            ret.y += elements[i];
    }

    return ret;
}

float3 get_vertex_position(sb_quadrilateral_patch patch, float2 tile_position)
{
    // Get basic vertex position on sphere from quad patch
    float3 position = patch.m_c00 +
                      patch.m_c10 * tile_position.x +
                      patch.m_c01 * tile_position.y +
                      patch.m_c11 * tile_position.x * tile_position.y +
                      patch.m_c20 * tile_position.x * tile_position.x +
                      patch.m_c02 * tile_position.y * tile_position.y +
                      patch.m_c21 * tile_position.x * tile_position.x * tile_position.y +
                      patch.m_c12 * tile_position.x * tile_position.y * tile_position.y;
    return position;
}

// Undefine math types
#ifdef __cplusplus
};

#include "mb_shared_types_undefine.hlsl"
#undef sb_quadrilateral_patch
#endif // __cplusplus

#endif // MB_SHADER_OCEAN_SHARED_COMMON
