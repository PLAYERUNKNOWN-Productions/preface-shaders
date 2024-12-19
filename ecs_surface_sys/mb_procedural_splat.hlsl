// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_procedural_common.hlsl"

// TODO: move to common
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_procedural_splat_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Utility
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Uses Slope and concave

float generate_tile_splat(float2 p_tile_coord,
                          float2 p_tile_uv,
                          float p_tile_size,
                          float p_elevation,
                          float p_terrain_slope,
                          uint p_tree_index,
                          uint l_splat_index)
{

    // Get cube coords
    float3 l_cube_coords = tile_to_cube_coords(p_tile_coord, p_tile_uv, p_tile_size, p_tree_index);

    // Map cube to sphere
    float3 l_sphere_coords = normalize(l_cube_coords);

    // For Djanco: Example mask usage
    float3 l_mask = load_mask(p_tile_uv,
                              g_push_constants.m_mask_buffer_srv,
                              g_push_constants.m_mask_channel_count,
                              g_push_constants.m_mask_resolution,
                              g_push_constants.m_elevation_resolution_x,
                              g_push_constants.m_elevation_resolution_y,
                              g_push_constants.m_elevation_border_width,
                              g_push_constants.m_tile_index);
    int l_biome_mask = trunc(l_mask.x);


//  [ For some reason the first material is empty in the quadtree gltf] //

    // Water Texture
    if(l_splat_index == 1)
    {
        return float(l_biome_mask == 0  // Ocean
                  || l_biome_mask == 16 // Lakes
                     );
    }
    // Forest ground texture
    else if(l_splat_index == 2)
    {
        return float(l_biome_mask == 1  // Tropical & Subtropical Moist Broadleaf Forests
                  || l_biome_mask == 2  // Tropical & Subtropical Dry Broadleaf Forests
                  || l_biome_mask == 3  // Tropical & Subtropical Coniferous Forests
                  || l_biome_mask == 4  // Temperate Broadleaf & Mixed Forests
                  || l_biome_mask == 5  // Temperate Conifer Forests
                  || l_biome_mask == 6  // Boreal Forests/Taiga
                  || l_biome_mask == 11 // Tundra
                     );
    }
    // Desert ground texture
    else if(l_splat_index == 3)
    {   
        return float(l_biome_mask == 7  // Tropical & Subtropical Grasslands, Savannas & Shrublands
                  || l_biome_mask == 12 // Mediterranean Forests, Woodlands & Scrub
                  || l_biome_mask == 13 // Deserts & Xeric Shrublands
                     );
    }
    // Grass ground texture
    else if(l_splat_index == 4)
    {
        return float(l_biome_mask == 8  // Temperate Grasslands, Savannas & Shrublands
                  || l_biome_mask == 10 // Montane Grasslands & Shrublands
                     );
    }
    // Snow texture
    else if(l_splat_index == 5)
    {
        return float(l_biome_mask == 15); // Ice
    }
    else
    {
        return 0.0;
    }

}

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------

[numthreads(PROCEDURAL_SPLAT_THREADGROUP_SIZE, PROCEDURAL_SPLAT_THREADGROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (p_dispatch_thread_id.x >= g_push_constants.m_resolution.x ||
        p_dispatch_thread_id.y >= g_push_constants.m_resolution.y)
    {
        return;
    }

    // Get resources
    RWStructuredBuffer<float> l_splat_buffer = ResourceDescriptorHeap[g_push_constants.m_buffer_uav];
    StructuredBuffer<float> l_elevation_buffer = ResourceDescriptorHeap[g_push_constants.m_elevation_buffer_srv];

    // Tile params
    float2 l_uv = pixel_coords_to_tile_uv(  p_dispatch_thread_id.xy,
                                            g_push_constants.m_resolution,
                                            g_push_constants.m_border_width);
    float2 l_fract_uv = frac(g_push_constants.m_tile_pos_frac_1000000.xy + l_uv.xy * g_push_constants.m_tile_pos_frac_1000000.zz);

    // Get elevation
    int2 l_elevation_resolution = int2(g_push_constants.m_elevation_resolution_x, g_push_constants.m_elevation_resolution_y);
#if !defined(ELEVATION_FROM_NOISE)
    float l_elevation = sample_elevation_linear_from_tile_uv(l_elevation_buffer, l_uv, l_elevation_resolution, g_push_constants.m_elevation_border_width, g_push_constants.m_elevation_offset);
#else
    float l_elevation = generate_tile_elevation_frac(l_fract_uv);
#endif


    // Derivative
#if !defined(ELEVATION_FROM_NOISE)
    float l_elevation_dx = sample_elevation_linear_from_tile_uv(l_elevation_buffer, l_uv + float2(1.0, 0.0) / g_push_constants.m_resolution, l_elevation_resolution, g_push_constants.m_elevation_border_width, g_push_constants.m_elevation_offset);
    float l_elevation_dy = sample_elevation_linear_from_tile_uv(l_elevation_buffer, l_uv + float2(0.0, 1.0) / g_push_constants.m_resolution, l_elevation_resolution, g_push_constants.m_elevation_border_width, g_push_constants.m_elevation_offset);
#else
    float l_elevation_dx = generate_tile_elevation_frac(l_fract_uv + float2(1.0, 0.0));
    float l_elevation_dy = generate_tile_elevation_frac(l_fract_uv + float2(0.0, 1.0));
#endif

    float l_slope = saturate((abs(l_elevation - l_elevation_dx) + abs(l_elevation - l_elevation_dy)) / (180000 * g_push_constants.m_tile_size));


    // Generate splat data
    uint l_pixel_offset = p_dispatch_thread_id.x + p_dispatch_thread_id.y * g_push_constants.m_resolution.x;
    l_splat_buffer[g_push_constants.m_tile_offset + 0 * g_push_constants.m_splat_offset + l_pixel_offset] = 1.0f;
    for (int l_index = 1; l_index < g_push_constants.m_splat_channels; ++l_index)
    {
        l_splat_buffer[g_push_constants.m_tile_offset + l_index * g_push_constants.m_splat_offset + l_pixel_offset] = generate_tile_splat(float2(g_push_constants.m_tile_x, g_push_constants.m_tile_y),
                                                                                                                                         l_uv,
                                                                                                                                         g_push_constants.m_tile_size,
                                                                                                                                         l_elevation,
                                                                                                                                         l_slope,
                                                                                                                                         g_push_constants.m_tree_index,
                                                                                                                                         l_index);
    }
}
