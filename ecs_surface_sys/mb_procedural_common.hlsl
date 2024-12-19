// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MB_SHADER_PROCEDURAL_COMMON
#define MB_SHADER_PROCEDURAL_COMMON

#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

// Enables elevation generation from noise
//#define ELEVATION_FROM_NOISE

//-----------------------------------------------------------------------------
float3 tile_to_cube_coords(float2 p_tile_coord,
                           float2 p_tile_uv,
                           float p_tile_size,
                           uint p_tree_index)
{
    // Get cube coords
    float2 l_quadtree_coords = (p_tile_coord + p_tile_uv * p_tile_size);

    // Get seed
    float3 l_cube_coords = 0;
    if (p_tree_index == 0)
    {
        l_cube_coords = float3(1.0f, l_quadtree_coords.y, l_quadtree_coords.x);
    }
    else if (p_tree_index == 1)
    {
        l_cube_coords = float3(-1.0f, l_quadtree_coords.y, -l_quadtree_coords.x);
    }
    else if (p_tree_index == 2)
    {
        l_cube_coords = float3(l_quadtree_coords.x, 1.0f, l_quadtree_coords.y);
    }
    else if (p_tree_index == 3)
    {
        l_cube_coords = float3(l_quadtree_coords.x, -1.0f, -l_quadtree_coords.y);
    }
    else if (p_tree_index == 4)
    {
        l_cube_coords = float3(-l_quadtree_coords.x, l_quadtree_coords.y, 1.0f);
    }
    else if (p_tree_index == 5)
    {
        l_cube_coords = float3(l_quadtree_coords.x, l_quadtree_coords.y, -1.0f);
    }

    return l_cube_coords;
}


//-----------------------------------------------------------------------------
// Noise based on fractured camera position
float generate_tile_elevation_frac(float2 p_uv)
{
    float noise = 0.0;
    int scale = 15;

    for(int i = 0; i <= 7; i++)
    {
        int scale_sqr = 2U << i;
        int final_scale = scale * scale_sqr;

        float tmp_noise = noise_repeat(p_uv * final_scale + i, final_scale);
        tmp_noise *= tmp_noise * tmp_noise;

        noise += tmp_noise / scale_sqr;
    }

    // return noise * noise * 70000.0;
    return noise * noise * noise * 200000.0;
}

//-----------------------------------------------------------------------------
float3 load_mask(float2 tile_uv,
                 uint mask_buffer_srv,
                 uint mask_channel_count,
                 uint mask_resolution,
                 uint elevation_resolution_x,
                 uint elevation_resolution_y,
                 uint elevation_border_width,
                 uint tile_index)
{
    if (mask_buffer_srv == RAL_NULL_BINDLESS_INDEX)
    {
        return (0.1f).xxx;
    }

    StructuredBuffer<float> mask_buffer = ResourceDescriptorHeap[mask_buffer_srv];

    const uint channel_count = mask_channel_count;
    const uint resolution = mask_resolution;
    const uint channel_stride = resolution * resolution;

    float2 mask_uv = tile_uv_to_texture_uv(tile_uv,
                                           uint2(elevation_resolution_x, elevation_resolution_y),
                                           elevation_border_width);

    uint base_offset = tile_index * resolution * resolution * channel_count;
    uint index_x = uint(mask_uv.x * resolution);
    uint index_y = uint(mask_uv.y * resolution);
    uint index = index_x + index_y * resolution;

    uint offset_index = base_offset + index;

    float3 mask = float3(mask_buffer[offset_index],
                         channel_count > 1 ? mask_buffer[offset_index + channel_stride] : 0.0f,
                         channel_count > 2 ? mask_buffer[offset_index + (channel_stride << 1)] : 0.0f);
    return mask;
}

#endif
