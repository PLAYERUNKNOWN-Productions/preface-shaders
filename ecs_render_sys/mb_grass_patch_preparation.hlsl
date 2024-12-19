// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"
#include "../ecs_surface_sys/mb_procedural_common.hlsl"
#include "mb_grass_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_grass_patch_preparation_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
uint4 get_tile_vertex_buffer_indices(uint2 vertex_index, uint tile_vertex_resolution, uint tile_index)
{
    uint4 indices;

    uint base_offset = tile_index * g_push_constants.m_tile_vertex_resolution * g_push_constants.m_tile_vertex_resolution;

    // 0, 0
    indices.x = vertex_index.x + vertex_index.y * tile_vertex_resolution + base_offset;

    // 1, 0
    indices.y = indices.x + 1;

    // 0, 1
    indices.z = indices.x + tile_vertex_resolution;

    // 1, 1
    indices.w = indices.z + 1;

    return indices;
}

//-----------------------------------------------------------------------------
float3 get_tile_data(StructuredBuffer<float3> buffer, uint4 indices, float2 lerp_factors)
{
    float3 data_00 = buffer[indices.x]; // tl
    float3 data_10 = buffer[indices.y]; // tr
    float3 data_01 = buffer[indices.z]; // bl
    float3 data_11 = buffer[indices.w]; // br

    float3 data = lerp(lerp(data_00, data_10, lerp_factors.x), lerp(data_01, data_11, lerp_factors.x), lerp_factors.y);

    return data;
}

//-----------------------------------------------------------------------------
uint hash(uint x) 
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

//-----------------------------------------------------------------------------
// Generates a [0, 1] float using a hash-based LCG
float rand(uint seed, uint index)
{
    // Mix seed and index
    seed ^= hash(index);
    seed ^= 0x9E3779B9u; // Constant used to disrupt linear sequences

    // Linear Congruential Generator (LCG)
    seed = seed * 1664525u + 1013904223u;

    // Convert to a float in the range [0, 1]
    return float(seed) / float(0xFFFFFFFFu);
}

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------

[numthreads(MB_GRASS_PATCH_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x >= g_push_constants.m_num_patches)
    {
        return;
    }

    StructuredBuffer<sb_grass_patch_preparation_t> preparation_buffer = ResourceDescriptorHeap[g_push_constants.m_preparation_buffer_srv];
    StructuredBuffer<float3> tile_position_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_position_buffer_srv];
    StructuredBuffer<float3> tile_normal_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_normal_buffer_srv];
    RWStructuredBuffer<sb_grass_patch_item_t> patch_buffer = ResourceDescriptorHeap[g_push_constants.m_patch_item_buffer_uav];

    sb_grass_patch_preparation_t prepared_data = preparation_buffer[dispatch_thread_id.x];

    // Tile UV
    float2 tile_position = prepared_data.m_uv * (1.0f / g_push_constants.m_tile_vertex_resolution);
    float2 tile_uv = tile_position_to_tile_uv(tile_position);

    uint random_seed = prepared_data.m_random_seed;

    float3 mask = load_mask(tile_uv,
                            g_push_constants.m_mask_buffer_srv,
                            g_push_constants.m_mask_channel_count,
                            g_push_constants.m_mask_resolution,
                            g_push_constants.m_elevation_resolution,
                            g_push_constants.m_elevation_resolution,
                            g_push_constants.m_elevation_border_width,
                            prepared_data.m_tile_index);
    uint biome_id = trunc(mask.x);
    // The tree coverage mask has a range of 0-100, so remap to 0-1
    float tree_coverage = saturate(mask.g / 100.0);

    float3 grass_biome_color = float3(0.31, 0.32, 0.13);
    float grass_biome_height = 0.3;
    // Lowering this value makes it that grass culls quicker when tree coverage value increases
    float grass_biome_spawning_biase = 1.0;

    switch(biome_id)
    {
        // Ocean
        case 0:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        // Tropical & Subtropical Grasslands, Savannas & Shrublands
        case 1:
            grass_biome_color = float3(0.640, 0.509, 0.205);
            grass_biome_height = 0.8;
            break;

        // Tropical & Subtropical Grasslands, Savannas & Shrublands
        case 2:
            grass_biome_color = float3(0.640, 0.509, 0.205);
            grass_biome_height = 0.8;
            break;

        // Tropical & Subtropical Coniferous Forests
        case 3:
            grass_biome_color = float3(0.144, 0.210, 0.11);
            grass_biome_height = 0.19;
            grass_biome_spawning_biase = 0.4;
            break;

        // Temperate Broadleaf & Mixed Forests
        case 4:
            grass_biome_color = float3(0.31, 0.32, 0.13);
            grass_biome_height = 0.2;
            grass_biome_spawning_biase = 0.55;
            break;

        // Tropical & Subtropical Coniferous Forests
        case 5:
            grass_biome_color = float3(0.144, 0.210, 0.11);
            grass_biome_height = 0.19;
            grass_biome_spawning_biase = 0.4;
            break;

        // Boreal Forests/Taiga
        case 6:
            grass_biome_color = float3(0.144, 0.210, 0.11);
            grass_biome_height = 0.19;
            break;

        // Tropical & Subtropical Grasslands, Savannas & Shrublands
        case 7:
            grass_biome_color = float3(0.640, 0.509, 0.205);
            grass_biome_height = 0.8;
            break;

        // 	Temperate Grasslands, Savannas & Shrublands
        case 8:
            grass_biome_color = float3(0.31, 0.32, 0.13);
            grass_biome_height = 0.2;
            break;

        // Flooded Grasslands & Savannas // Not supported
        case 9:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        // Montane Grasslands & Shrublands
        case 10:
            grass_biome_color = float3(0.31, 0.32, 0.13);
            grass_biome_height = 0.6;
            break;

        // Tundra
        case 11:
            grass_biome_color = float3(0.350, 0.143, 0.0140);
            grass_biome_height = 0.4;
            break;

        // Mediterranean Forests, Woodlands & Scrub // Not supported
        case 12:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        // Deserts & Xeric Shrublands
        case 13:
            grass_biome_color = float3(0.310, 0.277, 0.0930);
            grass_biome_height = 0.2;
            break;

        // Mangroves // Not supported
        case 14:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        // Ice, snow
        case 15:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        // Lakes  // Not supported
        case 16:
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;

        default:
            grass_biome_color = float3(0.31, 0.32, 0.13);
            break;

    }

    // Change grass density based on tree coverage value
    float rnd_cull_val = rand(random_seed, random_seed + random_seed);
    float grass_probability = tree_coverage / grass_biome_spawning_biase;
    // Low coverage values should not influence grass spawning probability to much, so powering the value
    grass_probability *= grass_probability;

    if (rnd_cull_val < grass_probability)
    {
            patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_INVALID;
            return;
    }

    // Scale grass based on tree coverage value
    grass_biome_height *= saturate(1.0 - grass_probability);

    // Get tile vertex buffer indices and lerp factors
    uint2 index = (uint2)prepared_data.m_uv; 
    float2 lerp_factors = prepared_data.m_uv - float2(index);
    uint4 indices = get_tile_vertex_buffer_indices(index, g_push_constants.m_tile_vertex_resolution, prepared_data.m_tile_index);

    // Get color from ground
    Texture2DArray vt0_array = ResourceDescriptorHeap[g_push_constants.m_tile_vt0_array_index_srv];
    float4 vt0 = vt0_array.SampleLevel( (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(tile_uv, prepared_data.m_tile_index), TEXTURE_ARRAY_MIP_LEVEL);

    float3 ground_normal = get_tile_data(tile_normal_buffer, indices, lerp_factors);

    // Add random offset
    float3 tangent = normalize(cross(ground_normal, float3(0, 0, 1)));
    float3 bitangent = normalize(cross(tangent, ground_normal));
    float3 patch_offset = (rand(random_seed, 25 - random_seed) - 0.5) * tangent + (rand(random_seed, 26 - random_seed) - 0.5) * bitangent;

    patch_buffer[dispatch_thread_id.x].m_type_id = GRASS_TYPE_ID_DEFAULT;
    patch_buffer[dispatch_thread_id.x].m_position_tile_local = get_tile_data(tile_position_buffer, indices, lerp_factors) + patch_offset;
    patch_buffer[dispatch_thread_id.x].m_ground_normal = ground_normal;
    patch_buffer[dispatch_thread_id.x].m_random_seed = random_seed;
    patch_buffer[dispatch_thread_id.x].m_ground_color = vt0.xyz;
    patch_buffer[dispatch_thread_id.x].m_color = grass_biome_color; // grass color per biome
    patch_buffer[dispatch_thread_id.x].m_blade_width = prepared_data.m_blade_width;
    patch_buffer[dispatch_thread_id.x].m_blade_height = grass_biome_height;
    patch_buffer[dispatch_thread_id.x].m_patch_radius = prepared_data.m_patch_radius;
}