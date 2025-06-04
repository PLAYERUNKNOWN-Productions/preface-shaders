// Copyright:   PlayerUnknown Productions BV

#ifndef MBSHADER_QUADTREE_COMMON_H
#define MBSHADER_QUADTREE_COMMON_H

#include "mb_common.hlsl"
#include "../shared_shaders/mb_shared_buffers.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"

#define TEXTURE_ARRAY_MIP_LEVEL 0

#define MB_QUADTREE_METALLIC_DEFAULT .1f

float4 get_color_on_rainbow(float p_v)
{
    p_v = saturate(p_v);
    float l_hue = 320.f - p_v * 320.f;
    float l_saturation = 1.0f;
    float l_lightness = 0.5f;
    float l_chroma = (1.f - abs(2.f * l_lightness - 1.f)) * l_saturation;
    float l_hue_prime = l_hue / 60.f;
    float l_hue_prime_trunc = trunc(l_hue_prime);
    float l_x = l_chroma * (1.f - abs(fmod(l_hue_prime, 2.f) - 1.f));
    float l_m = l_lightness - 0.5f * l_chroma;

    float3 l_rgb = (float3)0;
    if(l_hue_prime_trunc == 0.f)
    {
        l_rgb = float3(l_chroma, l_x, 0.f);
    }
    else if(l_hue_prime_trunc == 1.f)
    {
        l_rgb = float3(l_x, l_chroma, 0.f);
    }
    else if (l_hue_prime_trunc == 2.f)
    {
        l_rgb = float3(0.f, l_chroma, l_x);
    }
    else if (l_hue_prime_trunc == 3.f)
    {
        l_rgb = float3(0.f, l_x, l_chroma);
    }
    else if (l_hue_prime_trunc == 4.f)
    {
        l_rgb = float3(l_x, 0.f, l_chroma);
    }
    else
    {
        l_rgb = float3(l_chroma, 0.f, l_x);
    }

    return float4(l_rgb + l_m, 0);
}

struct terrain_sample_t
{
    float3 m_position_ws_local;
    float3 m_planet_normal_ws;
    float3 m_surface_normal_ws;
};

struct terrain_material_t
{
    float3 m_normal_ts;
    float3 m_base_color;
    float m_ao;
    float m_roughness;

    float m_debug_height;
    float2 m_tmp_uv;

};

// Convert from tile position to tile UV
float2 tile_position_to_tile_uv(float2 p_tile_position)
{
    return float2(p_tile_position.x, 1.0f - p_tile_position.y);
}

// Tile UV is in [0..1]
// Pixels centers should match with vertices
// Height has TILE_SIZE x TILE_SIZE pixels
// Patch has (TILE_SIZE - 1) x (TILE_SIZE - 1) quads
float2 tile_uv_to_texture_uv( float2 p_tile_uv,
                              float2 p_tile_resolution,
                              float p_tile_border_size)
{
    float2 l_resolution_without_border = p_tile_resolution - 2.0f * p_tile_border_size;
    return p_tile_uv * (l_resolution_without_border - 1.0f) / p_tile_resolution + (0.5f + p_tile_border_size) / p_tile_resolution;
}

// Converts coordinate in pixels to UV
// Pixels coorinates [0..tile_res + 2 * tile_border - 1]
// Pixel with a coordinate (tile_border - 1) is mapped ot UV 0
// Pixel with a coordinate (tile_res + tile_border - 1) is mapped ot UV 1
float2 pixel_coords_to_tile_uv( uint2 p_pixel_coords,
                                float2 p_tile_resolution,
                                float p_tile_border_size)
{
    float2 l_tile_position = ((int2)p_pixel_coords.xy - p_tile_border_size) / (p_tile_resolution - 2.0f * p_tile_border_size - 1.0);
    return tile_position_to_tile_uv(l_tile_position);
}

float sample_terrain_height(float2 p_tile_position,
                            uint2 p_heightmap_resolution,
                            uint p_heightmap_border,
                            Texture2DArray p_heightmap_array,
                            uint p_tile_index)
{
    float2 l_tile_uv = tile_position_to_tile_uv(p_tile_position);
    float2 l_height_uv = tile_uv_to_texture_uv(l_tile_uv, p_heightmap_resolution, p_heightmap_border);
    float l_height = p_heightmap_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(l_height_uv, p_tile_index), TEXTURE_ARRAY_MIP_LEVEL).x;

    return l_height;
}

float3 sample_terrain_position( sb_tile_instance_to_bake_t p_tile,
                                float2 p_tile_position,
                                float2 p_heightmap_resolution,
                                float p_heightmap_border,
                                Texture2DArray p_heightmap_array)
{
    // Get basic vertex position on sphere from quad patch
    float3 l_vertex = p_tile.m_patch_data.m_c00 +
                      p_tile.m_patch_data.m_c10 * p_tile_position.x +
                      p_tile.m_patch_data.m_c01 * p_tile_position.y +
                      p_tile.m_patch_data.m_c11 * p_tile_position.x * p_tile_position.y +
                      p_tile.m_patch_data.m_c20 * p_tile_position.x * p_tile_position.x +
                      p_tile.m_patch_data.m_c02 * p_tile_position.y * p_tile_position.y +
                      p_tile.m_patch_data.m_c21 * p_tile_position.x * p_tile_position.x * p_tile_position.y +
                      p_tile.m_patch_data.m_c12 * p_tile_position.x * p_tile_position.y * p_tile_position.y;

    // Interpolate normals
    float3 l_normal_01 = lerp(p_tile.m_normal_0, p_tile.m_normal_1, p_tile_position.x);
    float3 l_normal_23 = lerp(p_tile.m_normal_3, p_tile.m_normal_2, p_tile_position.x);
    float3 l_normal_ws = lerp(l_normal_01, l_normal_23, p_tile_position.y);
    l_normal_ws = normalize(l_normal_ws);

    // Fetch height
    float l_height = sample_terrain_height(p_tile_position, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile.m_tile_index);

    // Apply heightmap displacement
    l_vertex += l_height * l_normal_ws;

    return l_vertex;
}

float3 sample_terrain_normal(   sb_tile_instance_to_bake_t p_tile,
                                float2 p_tile_position,
                                float2 p_heightmap_resolution,
                                float p_heightmap_border,
                                Texture2DArray p_heightmap_array)
{
    // Compute terrain normal by sampling 4 points around
    float3 l_position_ws_0 = sample_terrain_position(p_tile, p_tile_position, p_heightmap_resolution, p_heightmap_border, p_heightmap_array);
    float3 l_position_ws_1 = sample_terrain_position(p_tile, p_tile_position + float2( 1.0f,  0.0f) / p_heightmap_resolution, p_heightmap_resolution, p_heightmap_border, p_heightmap_array);
    float3 l_position_ws_2 = sample_terrain_position(p_tile, p_tile_position + float2( 0.0f,  1.0f) / p_heightmap_resolution, p_heightmap_resolution, p_heightmap_border, p_heightmap_array);
    float3 l_position_ws_3 = sample_terrain_position(p_tile, p_tile_position + float2(-1.0f,  0.0f) / p_heightmap_resolution, p_heightmap_resolution, p_heightmap_border, p_heightmap_array);
    float3 l_position_ws_4 = sample_terrain_position(p_tile, p_tile_position + float2( 0.0f, -1.0f) / p_heightmap_resolution, p_heightmap_resolution, p_heightmap_border, p_heightmap_array);

    // Use cross-producs to avarage normals
    float3 l_normal_ws = 0;
    l_normal_ws += normalize(cross(l_position_ws_2 - l_position_ws_0, l_position_ws_1 - l_position_ws_0));
    l_normal_ws += normalize(cross(l_position_ws_1 - l_position_ws_0, l_position_ws_4 - l_position_ws_0));
    l_normal_ws += normalize(cross(l_position_ws_4 - l_position_ws_0, l_position_ws_3 - l_position_ws_0));
    l_normal_ws += normalize(cross(l_position_ws_3 - l_position_ws_0, l_position_ws_2 - l_position_ws_0));
    l_normal_ws = normalize(l_normal_ws);

    return l_normal_ws;
}

//! \brief Sample terrain normal in tile local space
float3 sample_terrain_normal(   float2 p_tile_position,
                                uint2 p_heightmap_resolution,
                                uint p_heightmap_border,
                                Texture2DArray p_heightmap_array,
                                uint p_tile_index,
                                float p_tile_size)
{
    // Compute terrain normal by sampling 4 points around
    float2 l_tile_position_0 = p_tile_position;
    float3 l_position_0 = float3(l_tile_position_0.x * p_tile_size,
                                 sample_terrain_height(l_tile_position_0, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile_index),
                                 l_tile_position_0.y * p_tile_size);

    float2 l_tile_position_1 = p_tile_position + float2( 1.0f,  0.0f) / p_heightmap_resolution;
    float3 l_position_1 = float3(l_tile_position_1.x * p_tile_size,
                                 sample_terrain_height(l_tile_position_1, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile_index),
                                 l_tile_position_1.y * p_tile_size);

    float2 l_tile_position_2 = p_tile_position + float2( 0.0f,  1.0f) / p_heightmap_resolution;
    float3 l_position_2 = float3(l_tile_position_2.x * p_tile_size,
                                 sample_terrain_height(l_tile_position_2, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile_index),
                                 l_tile_position_2.y * p_tile_size);

    float2 l_tile_position_3 = p_tile_position + float2(-1.0f,  0.0f) / p_heightmap_resolution;
    float3 l_position_3 = float3(l_tile_position_3.x * p_tile_size,
                                 sample_terrain_height(l_tile_position_3, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile_index),
                                 l_tile_position_3.y * p_tile_size);

    float2 l_tile_position_4 = p_tile_position + float2( 0.0f, -1.0f) / p_heightmap_resolution;
    float3 l_position_4 = float3(l_tile_position_4.x * p_tile_size,
                                 sample_terrain_height(l_tile_position_4, p_heightmap_resolution, p_heightmap_border, p_heightmap_array, p_tile_index),
                                 l_tile_position_4.y * p_tile_size);

    // Use cross-products to avarage normals
    float3 l_normal = 0;
    l_normal += normalize(cross(l_position_2 - l_position_0, l_position_1 - l_position_0));
    l_normal += normalize(cross(l_position_1 - l_position_0, l_position_4 - l_position_0));
    l_normal += normalize(cross(l_position_4 - l_position_0, l_position_3 - l_position_0));
    l_normal += normalize(cross(l_position_3 - l_position_0, l_position_2 - l_position_0));
    l_normal = normalize(l_normal);

    return l_normal;
}

float2 get_tile_position(sb_render_item_t p_render_item, uint p_index, uint p_vertex_resolution)
{
    // Get tile position
    uint l_tile_position_x = p_index % p_vertex_resolution;
    uint l_tile_position_y = (p_index - l_tile_position_x) / p_vertex_resolution;
    uint l_resolution_minus_one = p_vertex_resolution - 1;

    // This ternary expression is because, "l_tile_position_x / (float)l_resolution_minus_one" gives values that are not exactly 1 on AMD hardware (Radeon RX 6900 XT)
    float2 l_tile_position = (float2)0;
    l_tile_position.x = l_tile_position_x == l_resolution_minus_one ? 1.0f : (l_tile_position_x / (float)l_resolution_minus_one);
    l_tile_position.y = l_tile_position_y == l_resolution_minus_one ? 1.0f : (l_tile_position_y / (float)l_resolution_minus_one);

    return l_tile_position;
}

// Return data index in vertex buffer
uint get_tile_vertex_mesh_index(float2 p_tile_position, uint p_vertex_resolution)
{
    uint l_tile_position_x = round(p_tile_position.x * (p_vertex_resolution - 1));
    uint l_tile_position_y = round(p_tile_position.y * (p_vertex_resolution - 1));
    uint l_vertex_index = l_tile_position_y * p_vertex_resolution + l_tile_position_x;

    return l_vertex_index;
}

mesh_vertex_t get_tile_vertex_mesh(sb_render_item_t p_render_item, uint p_tile_instance_offset, uint p_vertex_mesh_index)
{
    // Init output
    mesh_vertex_t l_mesh_vertex = (mesh_vertex_t)0;

    // Get vertex mesh
    get_vertex_mesh_with_index(p_vertex_mesh_index, p_render_item, p_tile_instance_offset, l_mesh_vertex);

    return l_mesh_vertex;
}

terrain_sample_t sample_terrain(sb_render_item_t p_render_item, sb_tile_instance_t p_tile, float2 p_tile_position, uint p_vertex_resolution)
{
    // Default-init
    terrain_sample_t l_sample = (terrain_sample_t)0;

    // Interpolate normals
    float3 l_normal_01 = lerp(p_tile.m_basic_data.m_normal_0.xyz, p_tile.m_basic_data.m_normal_1.xyz, p_tile_position.x);
    float3 l_normal_23 = lerp(p_tile.m_basic_data.m_normal_3.xyz, p_tile.m_basic_data.m_normal_2.xyz, p_tile_position.x);
    float3 l_normal_ws = lerp(l_normal_01, l_normal_23, p_tile_position.y);
    l_normal_ws = normalize(l_normal_ws);

    // Get index from tile position
    uint l_vertex_mesh_index = get_tile_vertex_mesh_index(p_tile_position, p_vertex_resolution);

    // Get buffer offset for each tile instance
    uint l_instance_offset = p_tile.m_basic_data.m_tile_index * p_vertex_resolution * p_vertex_resolution;

    // Unpack mesh
    mesh_vertex_t l_mesh_vertex = get_tile_vertex_mesh(p_render_item, l_instance_offset, l_vertex_mesh_index);

    l_sample.m_position_ws_local   = l_mesh_vertex.m_position + p_tile.m_tile_local_to_camera_local;
    l_sample.m_planet_normal_ws    = l_normal_ws;
    l_sample.m_surface_normal_ws   = l_mesh_vertex.m_normal;

    return l_sample;
}

// Lerp mesh vertex
mesh_vertex_t lerp_mesh_vertex(mesh_vertex_t p_vertex_x,
                               mesh_vertex_t p_vertex_y,
                               float p_s)
{
    mesh_vertex_t l_vertex = (mesh_vertex_t)0;

    l_vertex.m_position = lerp(p_vertex_x.m_position, p_vertex_y.m_position, p_s);
    l_vertex.m_normal = lerp(p_vertex_x.m_normal, p_vertex_y.m_normal, p_s);

    return l_vertex;
}

// Sample terrain with linear filtering
terrain_sample_t sample_terrain_with_filtering(sb_render_item_t p_render_item, sb_tile_instance_t p_tile, float2 p_tile_position, uint p_vertex_resolution)
{
    terrain_sample_t l_sample = (terrain_sample_t)0;

    // Interpolate normals
    float3 l_normal_01 = lerp(p_tile.m_basic_data.m_normal_0.xyz, p_tile.m_basic_data.m_normal_1.xyz, p_tile_position.x);
    float3 l_normal_23 = lerp(p_tile.m_basic_data.m_normal_3.xyz, p_tile.m_basic_data.m_normal_2.xyz, p_tile_position.x);
    float3 l_normal_ws = lerp(l_normal_01, l_normal_23, p_tile_position.y);
    l_normal_ws = normalize(l_normal_ws);

    // Get buffer offset for each tile instance
    uint l_instance_offset = p_tile.m_basic_data.m_tile_index * p_vertex_resolution * p_vertex_resolution;

    // Get vertex indices
    float2 l_vertex_position = p_tile_position * (p_vertex_resolution - 1);
    uint2 l_vertex_position_floored = floor(l_vertex_position);
    uint2 l_vertex_position_ceiled = ceil(l_vertex_position);
    uint l_vertex_index_00 = l_vertex_position_floored.y * p_vertex_resolution + l_vertex_position_floored.x;
    uint l_vertex_index_01 = l_vertex_position_ceiled.y * p_vertex_resolution + l_vertex_position_floored.x;
    uint l_vertex_index_10 = l_vertex_position_floored.y * p_vertex_resolution + l_vertex_position_ceiled.x;
    uint l_vertex_index_11 = l_vertex_position_ceiled.y * p_vertex_resolution + l_vertex_position_ceiled.x;

    // Unpack mesh
    mesh_vertex_t l_vertex_00 = get_tile_vertex_mesh(p_render_item, l_instance_offset, l_vertex_index_00);
    mesh_vertex_t l_vertex_01 = get_tile_vertex_mesh(p_render_item, l_instance_offset, l_vertex_index_01);
    mesh_vertex_t l_vertex_10 = get_tile_vertex_mesh(p_render_item, l_instance_offset, l_vertex_index_10);
    mesh_vertex_t l_vertex_11 = get_tile_vertex_mesh(p_render_item, l_instance_offset, l_vertex_index_11);

    // Lerp
    mesh_vertex_t l_vertex_0 = lerp_mesh_vertex(l_vertex_00, l_vertex_01, l_vertex_position.y - l_vertex_position_floored.y);
    mesh_vertex_t l_vertex_1 = lerp_mesh_vertex(l_vertex_10, l_vertex_11, l_vertex_position.y - l_vertex_position_floored.y);
    mesh_vertex_t l_vertex = lerp_mesh_vertex(l_vertex_0, l_vertex_1, l_vertex_position.x - l_vertex_position_floored.x);

    l_sample.m_position_ws_local = l_vertex.m_position + p_tile.m_tile_local_to_camera_local;
    l_sample.m_planet_normal_ws = l_normal_ws;
    l_sample.m_surface_normal_ws = l_vertex.m_normal;

    return l_sample;
}

terrain_sample_t sample_terrain(sb_render_item_t p_render_item, sb_tile_instance_t p_tile, float2 p_tile_position, uint p_vertex_resolution, bool p_use_filtering)
{
    if (p_use_filtering)
    {
        return sample_terrain_with_filtering(p_render_item, p_tile, p_tile_position, p_vertex_resolution);
    }
    else
    {
        return sample_terrain(p_render_item, p_tile, p_tile_position, p_vertex_resolution);
    }
}

float3 sample_terrain_position(sb_render_item_t p_render_item, sb_tile_instance_t p_tile, float2 p_tile_position, uint p_vertex_resolution)
{
    // Get index from tile position
    uint l_vertex_mesh_index = get_tile_vertex_mesh_index(p_tile_position, p_vertex_resolution);

    // Get buffer offset for each tile instance
    uint l_instance_offset = p_tile.m_basic_data.m_tile_index * p_vertex_resolution * p_vertex_resolution;

    // Get position from buffer
    ByteAddressBuffer l_buffer = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_position_buffer_srv)];
    float3 l_position_tile_local = l_buffer.Load<float3>(l_instance_offset * p_render_item.m_position_stride + p_render_item.m_position_offset + l_vertex_mesh_index * p_render_item.m_position_stride);

    float3 l_position_camera_local = l_position_tile_local + p_tile.m_tile_local_to_camera_local;

    return l_position_camera_local;
}

terrain_material_t sample_terrain_material( uint p_tile_index,
                                            float2 p_tile_uv,
                                            uint p_tile_height_array_index_srv,
                                            uint p_tile_vt0_array_index_srv,
                                            uint p_tile_vt1_array_index_srv,
                                            uint2 p_heightmap_resolution,
                                            uint p_heightmap_border,
                                            uint2 p_vt_resolution,
                                            uint p_vt_border)
{
    terrain_material_t l_material = (terrain_material_t)0;

    // Textures
    Texture2DArray l_heightmap_array    = ResourceDescriptorHeap[NonUniformResourceIndex(p_tile_height_array_index_srv)];
    Texture2DArray l_vt0_array          = ResourceDescriptorHeap[NonUniformResourceIndex(p_tile_vt0_array_index_srv)];
    Texture2DArray l_vt1_array          = ResourceDescriptorHeap[NonUniformResourceIndex(p_tile_vt1_array_index_srv)];

    // UV
    float2 l_uv = tile_uv_to_texture_uv(p_tile_uv,
                                        p_heightmap_resolution,
                                        p_heightmap_border);

    // UV
    float2 l_uv_vt = tile_uv_to_texture_uv( p_tile_uv,
                                            p_vt_resolution,
                                            p_vt_border);

    // Read vt
    float4 l_vt0 = l_vt0_array.SampleLevel( (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                                            float3(l_uv_vt, p_tile_index),
                                            TEXTURE_ARRAY_MIP_LEVEL);
    float4 l_vt1 = l_vt1_array.SampleLevel( (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                                            float3(l_uv_vt, p_tile_index),
                                            TEXTURE_ARRAY_MIP_LEVEL);

    // Copy
    l_material.m_base_color     = gamma_to_linear(l_vt0.rgb);
    l_material.m_normal_ts.xy   = l_vt1.xy * 2.0f - 1.0f;
    l_material.m_normal_ts.z    = sqrt(1.0f - dot(l_material.m_normal_ts.xy, l_material.m_normal_ts.xy));
    l_material.m_roughness      = l_vt0.a;
    l_material.m_ao             = l_vt1.w;

    l_material.m_debug_height = l_heightmap_array.SampleLevel(  (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                                                                float3(l_uv, p_tile_index),
                                                                TEXTURE_ARRAY_MIP_LEVEL).x;

    l_material.m_tmp_uv = l_uv_vt;

    return l_material;
}

terrain_material_t sample_terrain_material( sb_tile_instance_t p_tile,
                                            sb_quadtree_material_t p_quadtree_material,
                                            float2 p_tile_uv,
                                            float3 p_position_ws_local)
{
    terrain_material_t l_material = sample_terrain_material(p_tile.m_basic_data.m_tile_index,
                                                            p_tile_uv,
                                                            p_quadtree_material.m_tile_height_array_index_srv,
                                                            p_quadtree_material.m_tile_vt0_array_index_srv,
                                                            p_quadtree_material.m_tile_vt1_array_index_srv,
                                                            p_quadtree_material.m_elevation_tile_resolution,
                                                            p_quadtree_material.m_elevation_tile_border,
                                                            p_quadtree_material.m_vt_resolution,
                                                            p_quadtree_material.m_vt_border);

    // Use ML generated satellite texture
    {
        Texture2DArray<float4> l_texturemap_array = ResourceDescriptorHeap[p_quadtree_material.m_tile_texture_array_index_srv];
        float2 l_uv = tile_uv_to_texture_uv(p_tile_uv,
                                            p_quadtree_material.m_elevation_tile_resolution,
                                            p_quadtree_material.m_elevation_tile_border);
        float4 l_texture = l_texturemap_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP],
                                                          float3(l_uv, p_tile.m_basic_data.m_tile_index),
                                                          TEXTURE_ARRAY_MIP_LEVEL);
        float3 l_remapped_color = l_texture.rgb / 255.0f;

        // Display Only satellite texture directly
        if (p_quadtree_material.m_splatmap_only)
        {
            // Already setup in l_material
        }
        else
        if (p_quadtree_material.m_ml_texture_only)
        {
            l_material.m_base_color = gamma_to_linear(l_remapped_color);
            l_material.m_roughness = 1;
        }
        else
        {
            // Distance blend with satellite and splatmap
            float l_blend_distance = p_quadtree_material.m_ml_texture_blend_distance;
            float l_blend_strength = p_quadtree_material.m_ml_texture_blend_strength;
            float l_blend_value = smoothstep(0, l_blend_distance, length(p_position_ws_local)) * (1.0f - l_blend_strength) + l_blend_strength;
            l_material.m_base_color = lerp(l_material.m_base_color, gamma_to_linear(l_remapped_color), l_blend_value);
            l_material.m_roughness = lerp(l_material.m_roughness, 0.9f, l_blend_value);
        }

        return l_material;
    }

    return l_material;
}

// Return true if p_tile_position changes to a value that does not match the exact vertex in buffer
void terrain_blend_mask_vertex( sb_tile_instance_t p_tile,
                                uint p_vertex_resolution,
                                inout float2 p_tile_position,
                                inout bool p_position_moved,
                                inout float p_blend_mask)
{
    // Merge vertices to match parent-tile vertex frequency
    // Round down each second vertex to match with parent-tile resolution
    uint l_res_minus_one = p_vertex_resolution - 1;
    uint2 l_vertex_int = p_tile_position.xy * l_res_minus_one + 0.5;
    l_vertex_int = (l_vertex_int / 2) * 2;

    float2 l_vertex = l_vertex_int / (float)l_res_minus_one;

    // Initialize values
    p_position_moved = false;
    p_blend_mask = 0;

    // To make sure we get a continuous terrain tile's edge and internal logic will be different
    // Edge: use edge factor
    // Internal: use tile's blend factor
    [flatten]
    if (any(p_tile_position == 0) ||
        any(p_tile_position == 1.0f))
    {
        // Edges

        // Left
        [flatten]
        if (p_tile.m_neighbours.x > 0 &&
            p_tile_position.x == 0)
        {
            p_tile_position.y = lerp(p_tile_position.y, l_vertex.y, p_tile.m_neighbours.x);
            p_position_moved |= (p_tile.m_neighbours.x > 0);
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours.x);
        }

        // Right
        [flatten]
        if (p_tile.m_neighbours.y > 0 &&
            p_tile_position.x == 1.0)
        {
            p_tile_position.y = lerp(p_tile_position.y, l_vertex.y, p_tile.m_neighbours.y);
            p_position_moved |= (p_tile.m_neighbours.y > 0);
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours.y);
        }

        // Botton
        [flatten]
        if( p_tile.m_neighbours.z > 0 &&
            p_tile_position.y == 0)
        {
            p_tile_position.x = lerp(p_tile_position.x, l_vertex.x, p_tile.m_neighbours.z);
            p_position_moved |= (p_tile.m_neighbours.z > 0);
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours.z);
        }

        // Top
        [flatten]
        if (p_tile.m_neighbours.w > 0 &&
            p_tile_position.y == 1.0)
        {
            p_tile_position.x = lerp(p_tile_position.x, l_vertex.x, p_tile.m_neighbours.w);
            p_position_moved |= (p_tile.m_neighbours.w > 0);
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours.w);
        }

        // Corners: no need to blend positions as corners never move!

        // Top-left
        [flatten]
        if (p_tile_position.x == 0 &&
            p_tile_position.y == 1.0)
        {
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours_diagonal.x);
        }

        // Top-right
        [flatten]
        if (p_tile_position.x == 1.0 &&
            p_tile_position.y == 1.0)
        {
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours_diagonal.y);
        }

        // Bottom-left
        [flatten]
        if (p_tile_position.x == 0 &&
            p_tile_position.y == 0)
        {
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours_diagonal.z);
        }

        // Bottom-right
        [flatten]
        if (p_tile_position.x == 1.0 &&
            p_tile_position.y == 0)
        {
            p_blend_mask = max(p_blend_mask, p_tile.m_neighbours_diagonal.w);
        }
    }
    else
    {
        // Tile's "internal part"
        p_tile_position.x = lerp(p_tile_position.x, l_vertex.x, p_tile.m_blend_to_parent);
        p_tile_position.y = lerp(p_tile_position.y, l_vertex.y, p_tile.m_blend_to_parent);
        p_position_moved |= (p_tile.m_blend_to_parent > 0);
        p_blend_mask = p_tile.m_blend_to_parent;
    }
}

float terrain_blend_mask_pixel( sb_tile_instance_t p_tile,
                                float2 p_tile_position,
                                float p_blend_range)
{
    // Initialize blend value
    float l_blend_mask = 0;
    float l_blend_range = saturate(p_blend_range);

    // To make sure we get a continuous terrain tile's edge and internal logic will be different
    // Edge: use edge factor
    // Internal: use tile's blend factor

    float2 l_mask_border = 1.0 - p_tile_position / l_blend_range;
    float2 l_mask_border_inv = 1.0 - (1.0 - p_tile_position) / l_blend_range;

    // Edges

    // Left, right, bottom, top
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours.x * l_mask_border.x);
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours.y * l_mask_border_inv.x);
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours.z * l_mask_border.y);
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours.w * l_mask_border_inv.y);

    // Corners

    // Top-left, Top-right, Bottom-left, Bottom-right
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours_diagonal.x * min(l_mask_border.x, l_mask_border_inv.y));
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours_diagonal.y * min(l_mask_border_inv.x, l_mask_border_inv.y));
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours_diagonal.z * min(l_mask_border.x, l_mask_border.y));
    l_blend_mask = max(l_blend_mask, p_tile.m_neighbours_diagonal.w * min(l_mask_border_inv.x, l_mask_border.y));

    return l_blend_mask;
}

//! \brief Sample elevation with point filtering
float sample_elevation_point(StructuredBuffer<float> p_buffer, int2 p_elevation_coords, int2 p_elevation_resolution, int p_elevation_data_offset)
{
    // Make sure we are not crossing data border
    p_elevation_coords.xy = clamp(p_elevation_coords.xy, 0, p_elevation_resolution - 1);

    uint l_elevation_pixel_offset = p_elevation_coords.x + p_elevation_coords.y * p_elevation_resolution.x;
    return p_buffer[p_elevation_data_offset + l_elevation_pixel_offset];
}

//! \brief Sample elevation with linear filtering
float sample_elevation_linear_from_tile_uv(StructuredBuffer<float> p_buffer, float2 p_tile_uv, int2 p_elevation_resolution, int p_elevation_border, int p_elevation_data_offset)
{
    float2 l_elevation_res = (float2)p_elevation_resolution;
    float2 l_elevation_coords = p_tile_uv * (l_elevation_res - 2.0f * p_elevation_border - 1.0f) + (float2)p_elevation_border;

    int2 l_elevation_coords_0 = ceil(l_elevation_coords - 0.5f);
    int2 l_elevation_coords_1 = ceil(l_elevation_coords - 0.5f) + int2(0, 1.0f);
    int2 l_elevation_coords_2 = ceil(l_elevation_coords - 0.5f) + int2(1.0f, 0);
    int2 l_elevation_coords_3 = ceil(l_elevation_coords - 0.5f) + int2(1.0f, 1.0f);

    float l_elevation_0 = sample_elevation_point(p_buffer, l_elevation_coords_0, p_elevation_resolution, p_elevation_data_offset);
    float l_elevation_1 = sample_elevation_point(p_buffer, l_elevation_coords_1, p_elevation_resolution, p_elevation_data_offset);
    float l_elevation_2 = sample_elevation_point(p_buffer, l_elevation_coords_2, p_elevation_resolution, p_elevation_data_offset);
    float l_elevation_3 = sample_elevation_point(p_buffer, l_elevation_coords_3, p_elevation_resolution, p_elevation_data_offset);

    float l_elevation_01 = lerp(l_elevation_0, l_elevation_1, l_elevation_coords.y - (float)l_elevation_coords_0.y);
    float l_elevation_23 = lerp(l_elevation_2, l_elevation_3, l_elevation_coords.y - (float)l_elevation_coords_0.y);
    float l_elevation = lerp(l_elevation_01, l_elevation_23, l_elevation_coords.x - (float)l_elevation_coords_0.x);

    return l_elevation;
}

void blend_with_parent( inout terrain_sample_t p_terrain_sample,
                        float2 p_tile_position,
                        sb_tile_instance_t p_tile,
                        sb_render_item_t p_render_item,
                        sb_tile_instance_t p_tile_parent,
                        uint p_vertex_resolution,
                        float p_blend_mask,
                        bool p_blend_with_parent)
{
    // Get position in parent's tile space
    float2 l_position_parent = p_tile_position * 0.5 + p_tile.m_parent_uv_offset;

    // Get mesh in parent's space
    terrain_sample_t l_sample_parent = sample_terrain(p_render_item, p_tile_parent, l_position_parent, p_vertex_resolution, p_blend_with_parent);

    // Blend
    p_terrain_sample.m_position_ws_local = lerp(p_terrain_sample.m_position_ws_local, l_sample_parent.m_position_ws_local, p_blend_mask);
    p_terrain_sample.m_planet_normal_ws = lerp(p_terrain_sample.m_planet_normal_ws, l_sample_parent.m_planet_normal_ws, p_blend_mask);
    p_terrain_sample.m_surface_normal_ws = lerp(p_terrain_sample.m_surface_normal_ws, l_sample_parent.m_surface_normal_ws, p_blend_mask);
}

void blend_with_parent(inout terrain_material_t p_material,
                       sb_tile_instance_t p_tile,
                       sb_quadtree_material_t p_quadtree_material,
                       float2 p_tile_position,
                       float p_blend_mask_vertex,
                       float3 p_position_ws_local)
{
    // Get patch data
    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(p_quadtree_material.m_tile_instance_buffer_index)];

    // Get parent material and blend
    if (p_tile.m_parent_index != TILE_NO_PARENT)
    {
        // Get parent's tile
        sb_tile_instance_t l_tile_parent = l_tile_instances[p_tile.m_parent_index];

        // Get position in parent's tile space
        float2 l_position_parent = p_tile_position * 0.5 + p_tile.m_parent_uv_offset;

        float2 l_tile_uv_parent = tile_position_to_tile_uv(l_position_parent);

        // Get blend_mask
        float l_blend_mask = terrain_blend_mask_pixel(p_tile, p_tile_position, p_quadtree_material.m_blend_range);
        l_blend_mask = max(l_blend_mask, p_blend_mask_vertex);

        terrain_material_t l_material_parent = sample_terrain_material(l_tile_parent, p_quadtree_material, l_tile_uv_parent, p_position_ws_local);
        p_material.m_base_color = lerp(p_material.m_base_color, l_material_parent.m_base_color, l_blend_mask);
        p_material.m_normal_ts  = lerp(p_material.m_normal_ts, l_material_parent.m_normal_ts, l_blend_mask);
        p_material.m_roughness  = lerp(p_material.m_roughness, l_material_parent.m_roughness, l_blend_mask);
        p_material.m_ao         = lerp(p_material.m_ao, l_material_parent.m_ao, l_blend_mask);
    }
}

//! \brief Apply skirt on tile borders to avoid cracks between tiles
void apply_skirt(float2 p_tile_position,
                 float p_skirt_distance_threshold_squared,
                 float p_skirt_scale,
                 sb_render_item_t p_render_item,
                 sb_tile_instance_t p_tile,
                 uint p_vertex_resolution,
                 inout float3 p_position_camera_local)
{
    // Only apply skirt on border vertices
    if (all(p_tile_position != 0) && all(p_tile_position != 1.0f))
    {
        return;
    }

    if (dot(p_position_camera_local, p_position_camera_local) > p_skirt_distance_threshold_squared)
    {
        return;
    }

    // Get the outward direction
    float2 l_inner_tile_position = (float2)0;
    [flatten]
    if ((p_tile_position.x == 0.0f || p_tile_position.x == 1.0f) && (p_tile_position.y == 0.0f || p_tile_position.y == 1.0f))
    {
        l_inner_tile_position = float2(0.5f, 0.5f);
    }
    else if (p_tile_position.x == 0.0f)
    {
        l_inner_tile_position = float2(1.0f, p_tile_position.y);
    }
    else if (p_tile_position.x == 1.0f)
    {
        l_inner_tile_position = float2(0.0f, p_tile_position.y);
    }
    else if (p_tile_position.y == 0.0f)
    {
        l_inner_tile_position = float2(p_tile_position.x, 1.0f);
    }
    else if (p_tile_position.y == 1.0f)
    {
        l_inner_tile_position = float2(p_tile_position.x, 0.0f);
    }
    float3 l_inner_position_camera_local = sample_terrain_position(p_render_item, p_tile, l_inner_tile_position, p_vertex_resolution);
    float3 l_skirt_dir = normalize(p_position_camera_local - l_inner_position_camera_local);

    // Update position
    p_position_camera_local += l_skirt_dir * p_skirt_scale;
}

void draw_tile_border(
    float2 position_local,
    inout float4 direct_lighting_output,
    inout float4 indirect_lighting_output)
{
    float border_size = 0.01;
    if (any(position_local < border_size) || any(position_local > 1.0 - border_size))
    {
        direct_lighting_output = (1.0 - direct_lighting_output);
        direct_lighting_output.r = 0.2;
        direct_lighting_output.gb *= 0.1;
        indirect_lighting_output = 0;
    }
}

void get_quadtree_debug_color(
    sb_quadtree_material_t quadtree_material,
    sb_tile_instance_t tile,
    terrain_material_t material,
    float2 position_local,
    float blend_mask,
    inout float4 direct_lighting_output,
    inout float4 indirect_lighting_output)
{
    if(quadtree_material.m_debug_terrain_mode != -1)
    {
        if (tile.m_basic_data.m_available)
        {
            if (quadtree_material.m_debug_terrain_mode == 0)//height
            {
                direct_lighting_output = float4(1, 0, 0, 0);

                if (quadtree_material.m_debug_sub_mode == 0)//gradient
                {
                    float h = clamp(material.m_debug_height, quadtree_material.m_height_debug_start, quadtree_material.m_height_debug_end);

                    float f = (h - quadtree_material.m_height_debug_start) / (quadtree_material.m_height_debug_end - quadtree_material.m_height_debug_start);
                    float3 start_color = (1.0f - f) * quadtree_material.m_height_debug_start_color;
                    float3 end_color = f * quadtree_material.m_height_debug_end_color;
                    direct_lighting_output = float4(start_color + end_color, 0);
                    indirect_lighting_output = 0;
                }
                else if(quadtree_material.m_debug_sub_mode == 1)//rainbow
                {
                    float f = (material.m_debug_height - quadtree_material.m_height_debug_start) / (quadtree_material.m_height_debug_end - quadtree_material.m_height_debug_start);
                    direct_lighting_output = get_color_on_rainbow(f);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 2)//fractional
                {
                    float g = frac(material.m_debug_height / quadtree_material.m_debug_terrain_gradient_scale);
                    direct_lighting_output = float4(g, g, g, 0);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 3)//fractional lines
                {
                    uint v = ((int)material.m_debug_height % quadtree_material.m_debug_terrain_gradient_scale) == 0;
                    direct_lighting_output = float4(v, v, v, 0);
                    indirect_lighting_output = 0;
                }
            }
            else if (quadtree_material.m_debug_terrain_mode == 2)//splat map
            {
                ByteAddressBuffer splat_buffer = ResourceDescriptorHeap[tile.m_splat_buffer_srv];

                float4 tensor_shape = quadtree_material.m_splat_shape;

                float2 tile_coord = float2(material.m_tmp_uv.x, material.m_tmp_uv.y) * tensor_shape.zw;
                float2 tile_uv = frac(tile_coord);

                uint4 offset_in_tile = trunc(float4(0, 0, tile_coord));
                uint offset =
                    offset_in_tile.x +
                    offset_in_tile.y * uint(tensor_shape.x) +
                    offset_in_tile.z * uint(tensor_shape.x * 1) +
                    offset_in_tile.w * uint(tensor_shape.x * 1 * tensor_shape.z);

                if (quadtree_material.m_debug_sub_mode == 0)//single
                {
                    float v = splat_buffer.Load<float>(4 * (tile.m_splat_data_offset + quadtree_material.m_debug_terrain_splat_channel * quadtree_material.m_splat_channel_offset + offset));
                    direct_lighting_output = float4(v, v, v, 0);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 1)//multi
                {
                    direct_lighting_output = (float4)0;
                    indirect_lighting_output = 0;

                    for (uint i = 0; i < quadtree_material.m_num_splat_map_channels_to_show; ++i)
                    {
                        uint channel = quadtree_material.m_splat_map_channels_to_show[i];
                        float factor = splat_buffer.Load<float>(4 * (tile.m_splat_data_offset + channel * quadtree_material.m_splat_channel_offset + offset));
                        direct_lighting_output += float4(quadtree_material.m_splat_map_channel_colors[i] * factor, 0);
                    }
                }
            }
            else if (quadtree_material.m_debug_terrain_mode == 3)//virtual textures
            {
                float2 tile_coord = float2(material.m_tmp_uv.x, material.m_tmp_uv.y);
                float2 tile_uv = frac(tile_coord);

                if (quadtree_material.m_debug_sub_mode == 0)//base color
                {
                    float2 uv_vt = tile_uv_to_texture_uv(tile_uv, quadtree_material.m_vt_resolution, quadtree_material.m_vt_border);
                    Texture2DArray vt_array = ResourceDescriptorHeap[quadtree_material.m_tile_vt0_array_index_srv];
                    float4 vt = vt_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(uv_vt, tile.m_basic_data.m_tile_index), TEXTURE_ARRAY_MIP_LEVEL);
                    direct_lighting_output = float4(gamma_to_linear(vt.rgb), 0);
                    indirect_lighting_output = 0;

                }
                else if (quadtree_material.m_debug_sub_mode == 1)//roughness
                {
                    float2 uv_vt = tile_uv_to_texture_uv(tile_uv, quadtree_material.m_vt_resolution, quadtree_material.m_vt_border);
                    Texture2DArray vt_array = ResourceDescriptorHeap[quadtree_material.m_tile_vt0_array_index_srv];
                    float4 vt = vt_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(uv_vt, tile.m_basic_data.m_tile_index), TEXTURE_ARRAY_MIP_LEVEL);
                    direct_lighting_output = float4(vt.aaa, 0);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 2)//normal
                {
                    float2 uv_vt = tile_uv_to_texture_uv(tile_uv, quadtree_material.m_vt_resolution, quadtree_material.m_vt_border);
                    Texture2DArray vt_array = ResourceDescriptorHeap[quadtree_material.m_tile_vt1_array_index_srv];
                    float4 vt = vt_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(uv_vt, tile.m_basic_data.m_tile_index), TEXTURE_ARRAY_MIP_LEVEL);
                    direct_lighting_output = float4(vt.rgb, 0);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 3)//ao
                {
                    float2 uv_vt = tile_uv_to_texture_uv(tile_uv, quadtree_material.m_vt_resolution, quadtree_material.m_vt_border);
                    Texture2DArray vt_array = ResourceDescriptorHeap[quadtree_material.m_tile_vt1_array_index_srv];
                    float4 vt = vt_array.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(uv_vt, tile.m_basic_data.m_tile_index), TEXTURE_ARRAY_MIP_LEVEL);
                    direct_lighting_output = float4(vt.aaa, 0);
                    indirect_lighting_output = 0;
                }
            }
            else if (quadtree_material.m_debug_terrain_mode == 4)//basemap debug
            {
                if (quadtree_material.m_debug_sub_mode == 0)//tile id
                {
                    float2 tile_uv = tile_position_to_tile_uv(position_local);
                    float2 tex_uv = tile_uv_to_texture_uv(
                        tile_uv,
                        quadtree_material.m_texture_tile_resolution,
                        quadtree_material.m_texture_tile_border);

                    StructuredBuffer<float> buffer = ResourceDescriptorHeap[quadtree_material.m_tensor_debug_buffer_srv];

                    const uint channel_count = quadtree_material.m_tensor_debug_channel_count;
                    const uint resolution = quadtree_material.m_tensor_debug_buffer_resolution;
                    const uint channel_stride = resolution * resolution;

                    uint base_offset = tile.m_basic_data.m_tile_index * resolution * resolution * channel_count;
                    uint index_x = uint(tex_uv.x * resolution);
                    uint index_y = uint(tex_uv.y * resolution);
                    uint index = index_x + index_y * resolution;

                    uint offset_index = base_offset + index;
                    float float_id = buffer[offset_index];
                    uint id = uint(float_id) & 0x3f; // We can only visualize up to the number 63 so we mask out the bits that we can show

                    float u_step = rcp(64.f);
                    float u_start = u_step * id;
                    float u_end = u_step * (id + 1);

                    float u = lerp(u_start, u_end, tile_uv.x);
                    float v = tile_uv.y;

                    float2 sample_uv = float2(u, v);

                    float3 color = bindless_tex2d_sample_level(quadtree_material.m_debug_texture_for_ints, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], sample_uv).rgb;

                    direct_lighting_output = float4(color, 0) * quadtree_material.m_tensor_visualizer_scalar;
                    indirect_lighting_output = float4(color, 0) * quadtree_material.m_tensor_visualizer_scalar;
                }
                else if (quadtree_material.m_debug_sub_mode == 1)//map to color
                {
                    float2 tile_uv = tile_position_to_tile_uv(position_local);
                    float2 uv = tile_uv_to_texture_uv(
                        tile_uv,
                        quadtree_material.m_texture_tile_resolution,
                        quadtree_material.m_texture_tile_border);

                    StructuredBuffer<float> buffer = ResourceDescriptorHeap[quadtree_material.m_tensor_debug_buffer_srv];

                    const uint channel_count = quadtree_material.m_tensor_debug_channel_count;
                    const uint resolution = quadtree_material.m_tensor_debug_buffer_resolution;
                    const uint channel_stride = resolution * resolution;

                    uint base_offset = tile.m_basic_data.m_tile_index * resolution * resolution * channel_count;
                    uint index_x = uint(uv.x * resolution);
                    uint index_y = uint(uv.y * resolution);
                    uint index = index_x + index_y * resolution;

                    uint offset_index = base_offset + index;
                    float3 texture = float3(buffer[offset_index],
                                              channel_count > 1 ? buffer[offset_index + channel_stride] : 0.0f,
                                              channel_count > 2 ? buffer[offset_index + (channel_stride << 1)] : 0.0f);

                    float3 tensor_debug_color = (float3)0;
                    uint channel_color_idx = 0;

                    float3 default_colors[3] = { float3(1,0,0), float3(0,1,0), float3(0,0,1) };

                    for(uint channel_idx = 0; channel_idx < 3; ++channel_idx)
                    {
                        uint channel_mask = 1u << channel_idx;

                        if((quadtree_material.m_tensor_debug_channel_mask & channel_mask) == 0)
                        {
                            texture[channel_idx] = 0.f;
                        }

                        if(quadtree_material.m_debug_sub_mode == 1)
                        {
                            tensor_debug_color += quadtree_material.m_tensor_visualizer_channel_color[channel_color_idx++] * texture[channel_idx];
                        }
                        else
                        {
                            tensor_debug_color[channel_idx] = texture[channel_idx];
                        }
                    }

                    direct_lighting_output = float4(tensor_debug_color, 0) * quadtree_material.m_tensor_visualizer_scalar;
                    indirect_lighting_output = float4(tensor_debug_color, 0) * quadtree_material.m_tensor_visualizer_scalar;
                }
            }
            else if (quadtree_material.m_debug_terrain_mode == 5)//terrain
            {
                if (quadtree_material.m_debug_sub_mode == 0)//tile id
                {
                    // Get default background color
                    const float3 bg_colors[] = {float3(1.0f, 0.5f, 0.5f), float3(1.0f, 1.0f, 0.5f), float3(0.5f, 0.5f, 1.0f), float3(0.5f, 1.0f, 0.5f)};
                    uint bg_color_index = tile.m_tile_id_xy.x % 2 + (tile.m_tile_id_xy.y % 2) * 2;
                    float3 color = bg_colors[bg_color_index];

                    const float2 uv_size_per_digit = float2(0.1f, 0.25f);
                    float2 org_uv = float2(position_local.x, 1.0f - position_local.y);

                    uint2 curr_grid_index = 0;
                    uint number_to_show = 0;
                    if (org_uv.y <= 0.25)
                    {
                        curr_grid_index.y = 0;
                        number_to_show = tile.m_quadtree_index;
                    }
                    else if (org_uv.y <= 0.5)
                    {
                        curr_grid_index.y = 1;
                        number_to_show = tile.m_basic_data.m_tile_level;
                    }
                    else if (org_uv.y <= 0.75)
                    {
                        curr_grid_index.y = 2;
                        number_to_show = tile.m_tile_id_xy.x;
                    }
                    else if (org_uv.y <= 1.0f)
                    {
                        curr_grid_index.y = 3;
                        number_to_show = tile.m_tile_id_xy.y;
                    }
                    curr_grid_index.x = (uint)(org_uv.x / uv_size_per_digit.x);

                    uint digit_index = 9 - min(curr_grid_index.x, 9);
                    uint multiple = 1;
                    for (uint i = 0; i < digit_index; i++)
                    {
                        multiple *= 10;
                    }
                    uint digit_to_show = (number_to_show / multiple) % 10;

                    if (number_to_show >= multiple || multiple == 1)
                    {
                        float2 curr_uv_in_grid = 0;
                        curr_uv_in_grid.x = (org_uv.x - curr_grid_index.x * uv_size_per_digit.x) / uv_size_per_digit.x;
                        curr_uv_in_grid.y = (org_uv.y - curr_grid_index.y * uv_size_per_digit.y) / uv_size_per_digit.y;

                        const float step = 1.0f / 64.0f;
                        float uv_u_start = step * digit_to_show;
                        float uv_u_end = step * (digit_to_show + 1);

                        float uv_u = lerp(uv_u_start, uv_u_end, curr_uv_in_grid.x);
                        float uv_v = curr_uv_in_grid.y;
                        float2 uv = float2(uv_u, uv_v);

                        color = bindless_tex2d_sample_level(quadtree_material.m_debug_texture_for_ints, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], uv).rgb;
                    }

                    direct_lighting_output = float4(color, 0);
                    indirect_lighting_output = 0;
                }
                else if (quadtree_material.m_debug_sub_mode == 1)//blend mask
                {
                    direct_lighting_output = float4(blend_mask, 0, 0, 0);
                    indirect_lighting_output = 0;
                }
            }
        }
    }

    if (quadtree_material.m_tile_border_enabled > 0)
    {
        draw_tile_border(position_local, direct_lighting_output, indirect_lighting_output);
    }
}

struct terrain_vertex_t
{
    float3 m_position_ws_local;
    float3 m_normal;
};

terrain_vertex_t get_terrain_vertex(sb_tile_instance_base p_tile, float2 p_tile_position, uint p_tile_height_array_index_srv, uint2 p_elevation_tile_resolution, uint p_elevation_tile_border)
{
    terrain_vertex_t l_mesh_vertex = (terrain_vertex_t)0;

    // Interpolate normals
    float3 l_normal_01 = lerp(p_tile.m_normal_0.xyz, p_tile.m_normal_1.xyz, p_tile_position.x);
    float3 l_normal_23 = lerp(p_tile.m_normal_3.xyz, p_tile.m_normal_2.xyz, p_tile_position.x);
    float3 l_normal_ws = lerp(l_normal_01, l_normal_23, p_tile_position.y);
    l_normal_ws = normalize(l_normal_ws);

    // Quad patch
    float3 l_point = p_tile.m_patch_data.m_c00.xyz +
                     p_tile.m_patch_data.m_c10.xyz * p_tile_position.x +
                     p_tile.m_patch_data.m_c01.xyz * p_tile_position.y +
                     p_tile.m_patch_data.m_c11.xyz * p_tile_position.x * p_tile_position.y +
                     p_tile.m_patch_data.m_c20.xyz * p_tile_position.x * p_tile_position.x +
                     p_tile.m_patch_data.m_c02.xyz * p_tile_position.y * p_tile_position.y +
                     p_tile.m_patch_data.m_c21.xyz * p_tile_position.x * p_tile_position.x * p_tile_position.y +
                     p_tile.m_patch_data.m_c12.xyz * p_tile_position.x * p_tile_position.y * p_tile_position.y;

    // TODO: can be moved to tile generation shader!
    // Vertex offset
    Texture2DArray l_heightmap_array = ResourceDescriptorHeap[p_tile_height_array_index_srv];
    float l_height = sample_terrain_height(p_tile_position, p_elevation_tile_resolution, p_elevation_tile_border, l_heightmap_array, p_tile.m_tile_index);
    l_point += l_height * l_normal_ws;

    l_mesh_vertex.m_position_ws_local = l_point;
    l_mesh_vertex.m_normal            = l_normal_ws;

    return l_mesh_vertex;
}

#endif // MBSHADER_QUADTREE_COMMON_H
