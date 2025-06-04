// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

#include "mb_procedural_common.hlsl"

#define MB_GDC_DEMO_PLACEMENT
//#define MB_PATHTRACING_PLACEMENT
//#define MB_GAMESCOM_PLACEMENT

// Push constants
ConstantBuffer<cb_procedural_population_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

void write_population_item(uint2 index, uint id, float3 offset, float scale, float rotation)
{
    uint l_population_item_offset = index.x + index.y * g_push_constants.m_population_dimension;

    RWStructuredBuffer<sb_population_item_t> l_population_buffer = ResourceDescriptorHeap[g_push_constants.m_population_buffer_uav];
    l_population_buffer[l_population_item_offset].m_item_id   = id;
    l_population_buffer[l_population_item_offset].m_offset.xz = (offset.xz + index) / (float)g_push_constants.m_population_dimension;
    l_population_buffer[l_population_item_offset].m_offset.y  = offset.y;
    l_population_buffer[l_population_item_offset].m_scale     = scale;
    l_population_buffer[l_population_item_offset].m_rotation  = rotation;
}

static const int BIOME_SPAWN_DATA_STRUCT_SIZE = 18;

struct biome_spawn_data
{
    uint m_id[BIOME_SPAWN_DATA_STRUCT_SIZE];
    int m_tile_level_limit[BIOME_SPAWN_DATA_STRUCT_SIZE];       // This is the tile size limit, this should make sure you can set a limit how far assets can spawn
    int m_tile_level_max_limit[BIOME_SPAWN_DATA_STRUCT_SIZE];   // This is the max tile size, with this we can remove large assets from spawning in small tiles causing pop ins
    float2 m_spawn_rate[BIOME_SPAWN_DATA_STRUCT_SIZE];          // interpolate between channel x and y based on coverage value
    float4 m_scale[BIOME_SPAWN_DATA_STRUCT_SIZE];               // x & z are min scale, y & w max scale, interpolate between xy and zw based on coverage value
    float2 m_scale_blend_exp[BIOME_SPAWN_DATA_STRUCT_SIZE];     // Exponent the interpolation value is powered with, this can be used to make larges scales more rare
    float4 m_z_offset[BIOME_SPAWN_DATA_STRUCT_SIZE];            // Same as ^ but then for height offset
};

void initialize_biome_spawn_data(inout biome_spawn_data biome_struct)
{
    for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
    {
        biome_struct.m_id[i]                   = 0u;
        biome_struct.m_tile_level_limit[i]     = 17;
        biome_struct.m_tile_level_max_limit[i] = 25;
        biome_struct.m_spawn_rate[i]           = float2(0.0, 0.0);
        biome_struct.m_scale[i]                = float4(0.8, 1.2, 0.8, 1.2);
        biome_struct.m_scale_blend_exp[i]      = float2(1.0, 1.0);
        biome_struct.m_z_offset[i]             = float4(0.0, 0.0, 0.0, 0.0);
    }
}

void normalize_biome_spawn_rate(inout biome_spawn_data biome_struct)
{
    float2 sum = float2(0.0, 0.0);

    for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
    {
        sum += biome_struct.m_spawn_rate[i];
    }

    // Only normalize if they spawn more than available
    if (sum.x > 1.0)
    {
        for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
        {
            biome_struct.m_spawn_rate[i].x /= sum.x;
        }
    }
    if (sum.y > 1.0)
    {
        for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
        {
            biome_struct.m_spawn_rate[i].y /= sum.y;
        }
    }
}

// Interpolates values based on coverage value. fist channel(s) will be replaced with result
void interpolate_data_on_coverage(inout biome_spawn_data biome_struct, float coverage)
{
    for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
    {
        biome_struct.m_spawn_rate[i].x      = lerp(biome_struct.m_spawn_rate[i].x,      biome_struct.m_spawn_rate[i].y,      coverage);
        biome_struct.m_scale[i].xy          = lerp(biome_struct.m_scale[i].xy,          biome_struct.m_scale[i].zw,          coverage);
        biome_struct.m_scale_blend_exp[i].x = lerp(biome_struct.m_scale_blend_exp[i].x, biome_struct.m_scale_blend_exp[i].y, coverage);
        biome_struct.m_z_offset[i].xy       = lerp(biome_struct.m_z_offset[i].xy,       biome_struct.m_z_offset[i].zw,       coverage);
    }
}

// If trees get culled in a small tile, we should increase its population chance to compensate
void compensate_max_tile_cull(inout biome_spawn_data biome_struct, uint biome_density, uint current_tile_level)
{
    for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
    {
        // The asset max tile size limit
        int asset_max_tile_level = biome_struct.m_tile_level_max_limit[i];
        // This value represents a gradient in tiles from asset_max_tile_level with the length of tile_gradient_level
        const uint tile_gradient_level = 2u;
        // This is the base value that gets powered
        const uint base_compensation_val = 2u;

        // Get the difference between asset tile max, and general biome max tile. This will tell us how many tile levels a asset will skip compared to the rest.
        int cull_tile_distance = biome_density - asset_max_tile_level;
        // Get the tile size difference between asset max tile level and the current tile level
        int tile_relative_level = asset_max_tile_level - current_tile_level;
        // Calculate gradient
        int asset_density_compensation = max(tile_gradient_level - tile_relative_level, 0);

        if (tile_relative_level >= 0 && tile_relative_level < tile_gradient_level)
        {
            // Calculate density difference using 3 tiles:
            // Original density       = a = 0.25^0 + 0.25^1 + 0.25^2
            // Density without tile 0 = b = 0.25^1 + 0.25^2
            // difference = a / b = 4.2
            biome_struct.m_spawn_rate[i] *= 4.2;
        }
    }
}

[numthreads(PROCEDURAL_POPULATION_THREADGROUP_SIZE, PROCEDURAL_POPULATION_THREADGROUP_SIZE, 1)]
void cs_main(uint2 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (any(p_dispatch_thread_id >= g_push_constants.m_population_dimension.xx))
    {
        return;
    }

    StructuredBuffer<float> l_elevation_buffer = ResourceDescriptorHeap[g_push_constants.m_elevation_buffer_srv];

    // Get tile UV
    float tile_size = g_push_constants.m_tile_size;
    float tile_x = g_push_constants.m_tile_x;
    float tile_y = g_push_constants.m_tile_y;
    float2 tile_coord = float2(tile_x, tile_y);

    // Default initialization
    write_population_item(p_dispatch_thread_id, 0, float3(0, 0, 0), 1, 0);


    float2 uv = pixel_coords_to_tile_uv(p_dispatch_thread_id, g_push_constants.m_population_dimension.xx, 0);
    float3 cube_coords = tile_to_cube_coords(tile_coord, uv, tile_size, g_push_constants.m_tree_index);
    float3 sphere_coords = normalize(cube_coords);

    // Unused currently, just fetched for TA to use
    float3 mask = load_mask(uv,
                              g_push_constants.m_mask_buffer_srv,
                              g_push_constants.m_mask_channel_count,
                              g_push_constants.m_mask_resolution,
                              g_push_constants.m_elevation_resolution_x,
                              g_push_constants.m_elevation_resolution_y,
                              g_push_constants.m_elevation_border_width,
                              g_push_constants.m_tile_index);

    uint biome_id = trunc(mask.x);
    // The tree coverage mask has a range of 0-100, so remap to 0-1
    float tree_coverage = saturate(mask.g / 100.0);

    float tile_coverage = g_push_constants.m_tile_coverage;

    // Terrain data
#if !defined(ELEVATION_FROM_NOISE)
    int2 elevation_resolution = int2(g_push_constants.m_elevation_resolution_x, g_push_constants.m_elevation_resolution_y);
    float height = sample_elevation_linear_from_tile_uv(l_elevation_buffer, uv, elevation_resolution, g_push_constants.m_elevation_border_width, g_push_constants.m_elevation_offset);
#else
    float height = generate_tile_elevation(tile_x, tile_y, uv, tile_size, g_push_constants.m_tree_index);
#endif


// Create biome data //

    biome_spawn_data biome_data;
    initialize_biome_spawn_data(biome_data);

    // Determin bome density
    uint biome_tile_density = 18;

    // When there is a float2 or a float4 there is a high chance the X and Y (for float4 XY and ZW) components are interpolated based on coverage


    // Ocean
    if (biome_id == 0)
    {
        return;
    }
    // Tropical & Subtropical Coniferous Forests and Temperate Conifer Forests
    else if (biome_id == 3 || biome_id == 5)
    {
        // Determins overall density of a biome
        biome_tile_density = 18;

        // Select assets id from pop_config.sqlite
        biome_data.m_id[0]  = 5;    // birch se1
        biome_data.m_id[1]  = 7;    // spruce l
        biome_data.m_id[2]  = 8;    // spruce m
        biome_data.m_id[3]  = 9;    // spruce se
        biome_data.m_id[4]  = 26;   // rock e
        biome_data.m_id[5]  = 25;   // rock b
        biome_data.m_id[6]  = 25;   // rock b (large)
        biome_data.m_id[7]  = 10;   // fern
        biome_data.m_id[8]  = 12;   // branch 01
        biome_data.m_id[9]  = 14;   // branch 05
        biome_data.m_id[10] = 15;   // branch 06
        biome_data.m_id[11] = 18;   // dead tree upright
        biome_data.m_id[12] = 19;   // dead tree fallen

        // Set spawning likelyhood. The sum can be lower than 1.0, but if the sum is larger than 1.0 the sum gets normalized to 1.0
        biome_data.m_spawn_rate[0]  = float2(0.0, 0.01);
        biome_data.m_spawn_rate[1]  = float2(0.0, 0.04);
        biome_data.m_spawn_rate[2]  = float2(0.0, 0.02);
        biome_data.m_spawn_rate[3]  = float2(0.07, 0.07);
        biome_data.m_spawn_rate[4]  = float2(0.05, 0.05);
        biome_data.m_spawn_rate[5]  = float2(0.05, 0.05);
        biome_data.m_spawn_rate[6]  = float2(0.01, 0.01);
        biome_data.m_spawn_rate[7]  = float2(0.8, 0.5);
        biome_data.m_spawn_rate[8]  = float2(0.0, 0.1);
        biome_data.m_spawn_rate[9]  = float2(0.0, 0.1);
        biome_data.m_spawn_rate[10] = float2(0.0, 0.1);
        biome_data.m_spawn_rate[11] = float2(0.0, 0.005);
        biome_data.m_spawn_rate[12] = float2(0.0, 0.005);

        biome_data.m_tile_level_limit[0]  = 16;
        biome_data.m_tile_level_limit[1]  = 10;
        biome_data.m_tile_level_limit[2]  = 12;
        biome_data.m_tile_level_limit[3]  = 16;
        biome_data.m_tile_level_limit[4]  = 17;
        biome_data.m_tile_level_limit[5]  = 17;
        biome_data.m_tile_level_limit[6]  = 12;
        biome_data.m_tile_level_limit[7]  = 17;
        biome_data.m_tile_level_limit[8]  = 17;
        biome_data.m_tile_level_limit[9]  = 16;
        biome_data.m_tile_level_limit[10] = 16;

        biome_data.m_tile_level_max_limit[1]  = 17;
        biome_data.m_tile_level_max_limit[2]  = 17;
        biome_data.m_tile_level_max_limit[6]  = 17;
        biome_data.m_tile_level_max_limit[11] = 17;

        // Set min and max scale per asset
        biome_data.m_scale[0]  = float2(0.7, 1.9).xyxy;
        biome_data.m_scale[1]  = float2(0.5, 1.3).xyxy;
        biome_data.m_scale[2]  = float2(0.5, 1.3).xyxy;
        biome_data.m_scale[3]  = float2(0.4, 1.1).xyxy;
        biome_data.m_scale[6]  = float2(2.0, 10.0).xyxy;
        biome_data.m_scale[7]  = float2(0.7, 1.4).xyxy;
        biome_data.m_scale[11] = float2(0.1, 1.0).xyxy;
        biome_data.m_scale[12] = float2(0.1, 1.0).xyxy;

        // The exponent the interpolation value is powered by, this can make large scale more rare
        biome_data.m_scale_blend_exp[3] = float2(3.0, 3.0);
        biome_data.m_scale_blend_exp[6] = float2(3.0, 3.0);

        // Z / height offset
        biome_data.m_z_offset[6] = float2(-2.5, 0.0).xyxy;
        biome_data.m_z_offset[7] = float2(-0.5, 0.0).xyxy;
    }
    // Broadleaf, mixed forest
    else if (biome_id == 4)
    {
        biome_data.m_id[0]  = 6;    // birch m1
        biome_data.m_id[1]  = 4;    // birch s2
        biome_data.m_id[2]  = 5;    // birch se1
        biome_data.m_id[3]  = 3;    // beech l1
        biome_data.m_id[4]  = 1;    // beech s1
        biome_data.m_id[5]  = 2;    // beech se1
        biome_data.m_id[6]  = 7;    // spruce l
        biome_data.m_id[7]  = 8;    // spruce m
        biome_data.m_id[8]  = 9;    // spruce se
        biome_data.m_id[9]  = 26;   // rock e
        biome_data.m_id[10] = 25;   // rock b
        biome_data.m_id[11] = 10;   // fern
        biome_data.m_id[12] = 11;   // blackberry bush
        biome_data.m_id[13] = 12;   // branch 01
        biome_data.m_id[14] = 14;   // branch 05
        biome_data.m_id[15] = 15;   // branch 06
        biome_data.m_id[16] = 18;   // dead tree upright
        biome_data.m_id[17] = 19;   // dead tree fallen

        biome_data.m_spawn_rate[0]  = float2(0.0, 0.025);
        biome_data.m_spawn_rate[1]  = float2(0.0, 0.025);
        biome_data.m_spawn_rate[2]  = float2(0.0, 0.25);
        biome_data.m_spawn_rate[3]  = float2(0.0, 0.025);
        biome_data.m_spawn_rate[4]  = float2(0.0, 0.1);
        biome_data.m_spawn_rate[5]  = float2(0.0, 0.25);
        biome_data.m_spawn_rate[6]  = float2(0.0, 0.025);
        biome_data.m_spawn_rate[7]  = float2(0.0, 0.025);
        biome_data.m_spawn_rate[8]  = float2(0.0, 0.1);
        biome_data.m_spawn_rate[9]  = float2(0.07, 0.07);
        biome_data.m_spawn_rate[10] = float2(0.07, 0.07);
        biome_data.m_spawn_rate[11] = float2(0.5, 0.5);
        biome_data.m_spawn_rate[12] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[13] = float2(0.0, 0.2);
        biome_data.m_spawn_rate[14] = float2(0.0, 0.2);
        biome_data.m_spawn_rate[15] = float2(0.0, 0.2);
        biome_data.m_spawn_rate[16] = float2(0.0, 0.01);
        biome_data.m_spawn_rate[17] = float2(0.0, 0.01);

        biome_data.m_tile_level_limit[0]  = 10;
        biome_data.m_tile_level_limit[1]  = 13;
        biome_data.m_tile_level_limit[2]  = 16;
        biome_data.m_tile_level_limit[3]  = 10;
        biome_data.m_tile_level_limit[4]  = 13;
        biome_data.m_tile_level_limit[5]  = 16;
        biome_data.m_tile_level_limit[6]  = 10;
        biome_data.m_tile_level_limit[7]  = 12;
        biome_data.m_tile_level_limit[8]  = 16;
        biome_data.m_tile_level_limit[9]  = 17;
        biome_data.m_tile_level_limit[10] = 17;
        biome_data.m_tile_level_limit[11] = 17;
        biome_data.m_tile_level_limit[12] = 17;
        biome_data.m_tile_level_limit[13] = 17;
        biome_data.m_tile_level_limit[14] = 17;
        biome_data.m_tile_level_limit[15] = 17;
        biome_data.m_tile_level_limit[16] = 15;
        biome_data.m_tile_level_limit[17] = 15;

        biome_data.m_tile_level_max_limit[0]  = 17;
        biome_data.m_tile_level_max_limit[3]  = 17;
        biome_data.m_tile_level_max_limit[6]  = 17;
        biome_data.m_tile_level_max_limit[7]  = 17;
        biome_data.m_tile_level_max_limit[16] = 17;

        biome_data.m_scale[6]  = float4(0.7, 1.0, 0.7, 1.0);
        biome_data.m_scale[7]  = float4(0.7, 1.0, 0.7, 1.0);
        biome_data.m_scale[8]  = float4(0.7, 1.0, 0.7, 1.0);
        biome_data.m_scale[10] = float4(0.3, 3.0, 0.3, 3.0);
        biome_data.m_scale[10] = float4(0.9, 1.8, 0.8, 1.5);
        biome_data.m_scale[13] = float4(0.5, 2.0, 0.5, 2.0);
        biome_data.m_scale[14] = float4(0.5, 1.5, 0.5, 1.5);
        biome_data.m_scale[15] = float4(0.5, 1.5, 0.5, 1.5);
        biome_data.m_scale[16] = float4(0.1, 1.0, 0.1, 1.0);
        biome_data.m_scale[17] = float4(0.1, 1.0, 0.1, 1.0);

        biome_data.m_scale_blend_exp[10] = float2(10.0, 10.0);
    }
    // Boreal Forests/Taiga
    else if (biome_id == 6)
    {
        biome_tile_density = 17;

        biome_data.m_id[0]  = 7;    // spruce l
        biome_data.m_id[1]  = 8;    // spruce m
        biome_data.m_id[2]  = 9;    // spruce se
        biome_data.m_id[3]  = 26;   // rock e
        biome_data.m_id[4]  = 25;   // rock b
        biome_data.m_id[5]  = 25;   // rock b (large)

        // Taiga biome has low tree coverage value (0-0.05)
        biome_data.m_spawn_rate[0] = float2(0.0, 0.08);
        biome_data.m_spawn_rate[1] = float2(0.0, 0.18);
        biome_data.m_spawn_rate[2] = float2(0.25, 0.3);
        biome_data.m_spawn_rate[3] = float2(0.05, 0.05);
        biome_data.m_spawn_rate[4] = float2(0.05, 0.05);
        biome_data.m_spawn_rate[5] = float2(0.01, 0.01);

        biome_data.m_tile_level_limit[0] = 10;
        biome_data.m_tile_level_limit[1] = 11;
        biome_data.m_tile_level_limit[2] = 14;
        biome_data.m_tile_level_limit[3] = 16;
        biome_data.m_tile_level_limit[4] = 16;
        biome_data.m_tile_level_limit[5] = 10;

        biome_data.m_tile_level_max_limit[0]  = 17;
        biome_data.m_tile_level_max_limit[1]  = 17;
        biome_data.m_tile_level_max_limit[5]  = 17;

        biome_data.m_scale[0] = float4(0.2, 0.7, 0.3, 1.2);
        biome_data.m_scale[1] = float4(0.2, 0.7, 0.3, 1.2);
        biome_data.m_scale[2] = float4(0.2, 0.7, 0.3, 1.5);
        biome_data.m_scale[3] = float2(0.6, 1.2).xyxy;
        biome_data.m_scale[4] = float2(0.4, 1.0).xyxy;
        biome_data.m_scale[5] = float2(3.0, 8.0).xyxy;

        biome_data.m_scale_blend_exp[0] = float2(3.0, 3.0);
        biome_data.m_scale_blend_exp[1] = float2(3.0, 3.0);
        biome_data.m_scale_blend_exp[2] = float2(5.0, 5.0);
        biome_data.m_scale_blend_exp[5] = float2(5.0, 5.0);

        biome_data.m_z_offset[3] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[4] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[5] = float2(-2.5, 0.0).xyxy;
    }
    // Tropical & Subtropical Grasslands, Savannas & Shrublands
    else if (biome_id == 7 || biome_id == 1 || biome_id == 2)
    {
        biome_data.m_id[0]  = 24;   // birch bush
        biome_data.m_id[1]  = 3;    // beech l (small)
        biome_data.m_id[2]  = 23;   // beech bush
        biome_data.m_id[3]  = 21;   // rock e
        biome_data.m_id[4]  = 20;   // rock b
        biome_data.m_id[5]  = 20;   // rock b (large)

        // Biome has low tree coverage value (0-0.05)
        biome_data.m_spawn_rate[0]  = float2(0.05, 0.03);
        biome_data.m_spawn_rate[1]  = float2(0.0, 0.1);
        biome_data.m_spawn_rate[2]  = float2(0.05, 0.02);
        biome_data.m_spawn_rate[3]  = float2(0.1, 0.1);
        biome_data.m_spawn_rate[4]  = float2(0.1, 0.1);
        biome_data.m_spawn_rate[5]  = float2(0.005, 0.005);

        biome_data.m_tile_level_limit[0]  = 16;
        biome_data.m_tile_level_limit[1]  = 13;
        biome_data.m_tile_level_limit[2]  = 16;
        biome_data.m_tile_level_limit[3]  = 17;
        biome_data.m_tile_level_limit[4]  = 17;
        biome_data.m_tile_level_limit[5]  = 11;

        biome_data.m_tile_level_max_limit[1]  = 17;
        biome_data.m_tile_level_max_limit[5]  = 17;

        biome_data.m_scale[0] = float2(0.9, 1.4).xyxy;
        biome_data.m_scale[1] = float4(0.2, 0.6, 0.3, 0.9);
        biome_data.m_scale[2] = float2(1.4, 1.9).xyxy;
        biome_data.m_scale[3] = float2(0.2, 0.5).xyxy;
        biome_data.m_scale[4] = float2(0.2, 0.5).xyxy;
        biome_data.m_scale[5] = float2(2.0, 7.0).xyxy;

        biome_data.m_scale_blend_exp[1] = float2(2.0, 2.0);
        biome_data.m_scale_blend_exp[5] = float2(10.0, 10.0);
    }
    // Temperate Grasslands, Savannas & Shrublands
    else if (biome_id == 8)
    {
        biome_data.m_id[0]  = 24;   // birch bush
        biome_data.m_id[1]  = 23;   // beech bush
        biome_data.m_id[2]  = 2;    // beech se1
        biome_data.m_id[3]  = 17;   // rock e
        biome_data.m_id[4]  = 16;   // rock b

        biome_data.m_spawn_rate[0] = float2(0.01, 0.01);
        biome_data.m_spawn_rate[1] = float2(0.035, 0.035);
        biome_data.m_spawn_rate[2] = float2(0.006, 0.006);
        biome_data.m_spawn_rate[3] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[4] = float2(0.1, 0.1);

        biome_data.m_tile_level_limit[0] = 16;
        biome_data.m_tile_level_limit[1] = 16;
        biome_data.m_tile_level_limit[2] = 15;
        biome_data.m_tile_level_limit[3] = 17;
        biome_data.m_tile_level_limit[4] = 17;

        biome_data.m_scale[0] = float2(0.9, 1.7).xyxy;
        biome_data.m_scale[1] = float2(0.9, 1.7).xyxy;
        biome_data.m_scale[2] = float2(0.4, 1.1).xyxy;
        biome_data.m_scale[3] = float2(0.2, 0.4).xyxy;
        biome_data.m_scale[4] = float2(0.2, 0.4).xyxy;

        biome_data.m_z_offset[3] = float2(-0.5, 0.0).xyxy;
        biome_data.m_z_offset[4] = float2(-0.5, 0.0).xyxy;
    }
    // Flooded Grasslands & Savannas (Not supported by ML)
    else if (biome_id == 9)
    {
        return;
    }
    // Montane Grasslands & Shrublands
    else if (biome_id == 10)
    {
        biome_data.m_id[0]  = 24;   // birch bush
        biome_data.m_id[1]  = 23;   // beech bush
        biome_data.m_id[2]  = 2;    // beech se1
        biome_data.m_id[3]  = 17;   // rock e
        biome_data.m_id[4]  = 16;   // rock b
        biome_data.m_id[5]  = 16;   // rock b (larger)

        biome_data.m_spawn_rate[0] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[1] = float2(0.02, 0.02);
        biome_data.m_spawn_rate[2] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[3] = float2(0.2, 0.2);
        biome_data.m_spawn_rate[4] = float2(0.2, 0.2);
        biome_data.m_spawn_rate[5] = float2(0.007, 0.007);

        biome_data.m_tile_level_limit[0] = 16;
        biome_data.m_tile_level_limit[1] = 16;
        biome_data.m_tile_level_limit[2] = 16;
        biome_data.m_tile_level_limit[3] = 16;
        biome_data.m_tile_level_limit[4] = 16;
        biome_data.m_tile_level_limit[5] = 10;

        biome_data.m_tile_level_max_limit[5]  = 17;

        biome_data.m_scale[0] = float2(0.8, 1.7).xyxy;
        biome_data.m_scale[1] = float2(0.8, 1.7).xyxy;
        biome_data.m_scale[2] = float2(0.3, 1.1).xyxy;
        biome_data.m_scale[3] = float2(0.4, 1.0).xyxy;
        biome_data.m_scale[4] = float2(0.4, 1.0).xyxy;
        biome_data.m_scale[5] = float2(3.0, 8.0).xyxy;

        biome_data.m_scale_blend_exp[5] = float2(10.0, 10.0);

        biome_data.m_z_offset[3] = float2(-0.5, 0.0).xyxy;
        biome_data.m_z_offset[4] = float2(-0.5, 0.0).xyxy;
        biome_data.m_z_offset[5] = float2(-2.5, 0.0).xyxy;
    }
    // Tundra
    else if (biome_id == 11)
    {
        biome_data.m_id[1]  = 9;    // spruce se1
        biome_data.m_id[2]  = 11;   // blackberry bush
        biome_data.m_id[3]  = 17;   // rock e
        biome_data.m_id[4]  = 16;   // rock b
        biome_data.m_id[5]  = 16;   // rock b (larger)

        biome_data.m_spawn_rate[1] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[2] = float2(0.07, 0.07);
        biome_data.m_spawn_rate[3] = float2(0.04, 0.04);
        biome_data.m_spawn_rate[4] = float2(0.04, 0.04);
        biome_data.m_spawn_rate[5] = float2(0.008, 0.008);

        biome_data.m_tile_level_limit[1] = 15;
        biome_data.m_tile_level_limit[2] = 16;
        biome_data.m_tile_level_limit[3] = 15;
        biome_data.m_tile_level_limit[4] = 15;
        biome_data.m_tile_level_limit[5] = 12;

        biome_data.m_tile_level_max_limit[5]  = 17;

        biome_data.m_scale[1] = float2(0.25, 0.8).xyxy;
        biome_data.m_scale[2] = float2(0.4, 1.2).xyxy;
        biome_data.m_scale[3] = float2(0.5, 1.2).xyxy;
        biome_data.m_scale[4] = float2(0.4, 1.0).xyxy;
        biome_data.m_scale[5] = float2(3.0, 8.0).xyxy;

        biome_data.m_scale_blend_exp[1] = float2(3.0, 3.0);
        biome_data.m_scale_blend_exp[5] = float2(5.0, 5.0);

        biome_data.m_z_offset[3] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[4] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[5] = float2(-2.5, 0.0).xyxy;
    }
    // Mediterranean Forests, Woodlands & Scrub (Not supported by ML)
    else if (biome_id == 12)
    {
        return;
    }
    // Deserts & Xeric Shrublands
    else if (biome_id == 13)
    {
        biome_tile_density = 19;

        biome_data.m_id[0] = 23; // birch bush
        biome_data.m_id[1] = 24; // beech bush
        biome_data.m_id[2] = 20; // rock desert b
        biome_data.m_id[3] = 21; // rock desert e
        biome_data.m_id[4] = 22; // cactus
        biome_data.m_id[5] = 20; // rock desert b (scaled larger)

        biome_data.m_tile_level_limit[0] = 15;
        biome_data.m_tile_level_limit[1] = 15;
        biome_data.m_tile_level_limit[2] = 15;
        biome_data.m_tile_level_limit[3] = 15;
        biome_data.m_tile_level_limit[4] = 15;
        biome_data.m_tile_level_limit[5] = 10;

        biome_data.m_spawn_rate[0] = float2(0.06, 0.06);
        biome_data.m_spawn_rate[1] = float2(0.06, 0.06);
        biome_data.m_spawn_rate[2] = float2(0.3, 0.3);
        biome_data.m_spawn_rate[3] = float2(0.3, 0.3);
        biome_data.m_spawn_rate[4] = float2(0.0025, 0.0025);
        biome_data.m_spawn_rate[5] = float2(0.0001, 0.0001);

        biome_data.m_tile_level_max_limit[4]  = 17;
        biome_data.m_tile_level_max_limit[5]  = 17;

        biome_data.m_scale[0] = float2(0.3, 1.4).xyxy;
        biome_data.m_scale[1] = float2(0.5, 3.0).xyxy;
        biome_data.m_scale[2] = float2(0.15, 2.5).xyxy;
        biome_data.m_scale[3] = float2(0.15, 2.5).xyxy;
        biome_data.m_scale[4] = float2(1.5, 2.3).xyxy;
        biome_data.m_scale[5] = float2(5.0, 15.0).xyxy;

        biome_data.m_z_offset[0] = float2(-0.25, 0.0).xyxy;
        biome_data.m_z_offset[1] = float2(-0.25, 0.0).xyxy;

        biome_data.m_scale_blend_exp[1] = float2(5.0, 5.0);
        biome_data.m_scale_blend_exp[2] = float2(50.0, 50.0);
        biome_data.m_scale_blend_exp[3] = float2(50.0, 50.0);
        biome_data.m_scale_blend_exp[5] = float2(3.0, 3.0);
    }
    // Mangroves (Not supported by ML)
    else if (biome_id == 14)
    {
        return;
    }
    // Ice
    else if (biome_id == 15)
    {
        biome_tile_density = 18;

        biome_data.m_id[0]  = 28;   // rock e snow
        biome_data.m_id[1]  = 27;   // rock b snow
        biome_data.m_id[2]  = 27;   // rock b snow (large)

        biome_data.m_tile_level_limit[0] = 15;
        biome_data.m_tile_level_limit[1] = 15;
        biome_data.m_tile_level_limit[2] = 0;

        biome_data.m_tile_level_max_limit[2]  = 17;

        biome_data.m_spawn_rate[0] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[1] = float2(0.1, 0.1);
        biome_data.m_spawn_rate[2] = float2(0.004, 0.004);

        biome_data.m_scale[0] = float2(0.7, 1.8).xyxy;
        biome_data.m_scale[1] = float2(0.3, 1.8).xyxy;
        biome_data.m_scale[2] = float2(4.0, 10.0).xyxy;

        biome_data.m_scale_blend_exp[0] = float2(5.0, 5.0);
        biome_data.m_scale_blend_exp[1] = float2(5.0, 5.0);
        biome_data.m_scale_blend_exp[2] = float2(15.0, 15.0);

        biome_data.m_z_offset[0] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[1] = float2(-0.1, 0.0).xyxy;
        biome_data.m_z_offset[2] = float2(-4.0, 0.0).xyxy;
    }
    // Lakes (Not supported by ML)
    else if (biome_id == 16)
    {
        return;
    }
    else
    {
        return;
    }

    // Interpolate values (like offset, spawnrate, offset, etc) based on coverage
    interpolate_data_on_coverage(biome_data, saturate(tree_coverage));
    // If trees get culled in a small tile, we should increase its population chance to compensate
    compensate_max_tile_cull(biome_data, biome_tile_density, g_push_constants.m_tile_level);
    // The sum can be lower than 1.0, but if the sum is larger than 1.0 the sum gets normalized to 1.0
    normalize_biome_spawn_rate(biome_data);

// End biome data //


    // Random values
    float spawnrate_val = hash_3(uint3(p_dispatch_thread_id + g_push_constants.m_tile_seed, 0u)).x;
    float3 rnd_val_01 =   hash_3(uint3(p_dispatch_thread_id * g_push_constants.m_tile_seed, 10u));

    // Initialize values
    uint id = 0;
    float scale = 1.0;
    float z_offset = 0.0;

    // Set overall biome density
    if (g_push_constants.m_tile_level > biome_tile_density)
    {
        return;
    }

    // Loop variable
    float spawnrate_sum = 0.0;

    for (int i = 0; i < BIOME_SPAWN_DATA_STRUCT_SIZE; i++)
    {
        // Dont spawn asset if tile limit has been reached. This makes it so we can exclude small assets like branches from far away/large tiles
        if (biome_data.m_tile_level_limit[i] >= g_push_constants.m_tile_level || biome_data.m_tile_level_max_limit[i] < g_push_constants.m_tile_level)
        {
            continue;
        }

        // If asset has ID of 0, skip the asset
        if (biome_data.m_id[i] == 0)
        {
            continue;
        }

        if (spawnrate_val < biome_data.m_spawn_rate[i].x + spawnrate_sum && spawnrate_val > spawnrate_sum)
        {
            id = biome_data.m_id[i];

            // get min and max scale
            scale = biome_data.m_scale[i].x + (biome_data.m_scale[i].y - biome_data.m_scale[i].x) * pow(rnd_val_01.x, biome_data.m_scale_blend_exp[i].x);

            // get min and max z offset
            z_offset = biome_data.m_z_offset[i].x + (biome_data.m_z_offset[i].y - biome_data.m_z_offset[i].x) * rnd_val_01.y;
        }
        spawnrate_sum += biome_data.m_spawn_rate[i].x;
    }

    // Random Values
    float2 offset = hash_3(uint3(p_dispatch_thread_id + g_push_constants.m_tile_seed, 2u)).xy;
    float3 rand_transform = hash_3(uint3(p_dispatch_thread_id * g_push_constants.m_tile_seed, 3u));
    float rand_rotation = 2.0f * M_PI * rand_transform.x;


    write_population_item(p_dispatch_thread_id,
                          id,
                          float3(offset.x, z_offset, offset.y),
                          scale,
                          rand_rotation);
}
