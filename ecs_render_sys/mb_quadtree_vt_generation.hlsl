// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

// Variants
// OUTPUT_UV

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_vt_generation_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct quadtree_material_t
{
    float3 m_base_color;
    float3 m_normal_ts;
    float m_ao;
    float m_roughness;
    float m_local_height;
};

//-----------------------------------------------------------------------------
quadtree_material_t get_material(   StructuredBuffer<sb_geometry_pbr_material_t> p_pbr_material_list,
                                    int p_index,
                                    float2 p_uv,
                                    int p_mip_level)
{
    quadtree_material_t l_material;

    //-------------------------------------------------------------------------
    // Procadural materials

    if (p_index == g_push_constants.m_material_count + 0)
    {
        l_material.m_base_color         = 1.0f;
        l_material.m_normal_ts          = float3(0.5f, 0.5f, 1.0f);
        l_material.m_ao                 = 1.0f;
        l_material.m_roughness          = 1.0f;
        l_material.m_local_height       = 1.0f;
        return l_material;
    }

    //-------------------------------------------------------------------------
    // Bound check
    if (p_index >= g_push_constants.m_material_count)
    {
        return (quadtree_material_t)0;
    }

    //-------------------------------------------------------------------------
    // Texture

    // Get material from the list
    sb_geometry_pbr_material_t l_pbr_material = p_pbr_material_list[NonUniformResourceIndex(p_index)];

    // Read textures
    float4 l_base_color_texture = bindless_tex2d_sample_level(NonUniformResourceIndex(l_pbr_material.m_base_color_texture_srv), (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], p_uv, p_mip_level, 0);
    float3 l_normal_texture     = bindless_tex2d_sample_level(NonUniformResourceIndex(l_pbr_material.m_normal_map_texture_srv), (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], p_uv, p_mip_level, 0).xyz;
    float4 l_mask_texture       = bindless_tex2d_sample_level(NonUniformResourceIndex(l_pbr_material.m_occlusion_texture_srv), (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], p_uv, p_mip_level, 0);

    // Unpack data
    l_material.m_base_color         = l_base_color_texture.rgb;
    l_material.m_normal_ts          = l_normal_texture;
    l_material.m_ao                 = l_mask_texture.r;
    l_material.m_roughness          = l_mask_texture.b;
    l_material.m_local_height       = l_mask_texture.g;

    return l_material;
}

//-----------------------------------------------------------------------------
float2 rotate_2D(float2 p_uv, float p_angle)
{
    float2x2 l_rot_matrix = float2x2(cos(p_angle), -sin(p_angle),
                                     sin(p_angle),  cos(p_angle));
    return mul(l_rot_matrix, p_uv);
}

//-----------------------------------------------------------------------------
// Height blend for vector
float3 height_blend(float3 p_color_01, float3 p_color_02, float p_heightmap_01, float p_heightmap_02, float2 p_mask)
{
    float2 l_heightmap = float2(p_heightmap_01, p_heightmap_02) * p_mask;

#if 0 //Debug renders blending colors
    return l_heightmap.r > l_heightmap.g ? (p_color_01 + float3(0.20,0,0)) / 1.2 : (p_color_02 + float3(0,0.20,0)) / 1.2;
#endif

    return l_heightmap.r > l_heightmap.g ? p_color_01 : p_color_02;
}

//-----------------------------------------------------------------------------
// Heightblend for greyscale
float height_blend(float p_color_01, float p_color_02, float p_heightmap_01, float p_heightmap_02, float2 p_mask)
{
    float2 l_heightmap = float2(p_heightmap_01, p_heightmap_02) * p_mask;
    return l_heightmap.r > l_heightmap.g ? p_color_01 : p_color_02;
}

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------

[numthreads(VT_GENERATION_THREADGROUP_SIZE, VT_GENERATION_THREADGROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_vt_resolution))
    {
        return;
    }

    // Get source materials
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[g_push_constants.m_material_buffer_srv];

    // Get output textures
    RWTexture2D<float4> l_vt0_tile = ResourceDescriptorHeap[g_push_constants.m_tile_vt0_tmp_uav];
    RWTexture2D<float4> l_vt1_tile = ResourceDescriptorHeap[g_push_constants.m_tile_vt1_tmp_uav];

    // Tile params
    float2 l_uv = pixel_coords_to_tile_uv(  p_dispatch_thread_id.xy,
                                            g_push_constants.m_vt_resolution,
                                            g_push_constants.m_vt_border);
    float2 l_tile_position = p_dispatch_thread_id.xy / (float2)(g_push_constants.m_vt_resolution);
    float2 l_tile_uv = tile_position_to_tile_uv(l_tile_position);

    // Scale UV to get UV for tiled textures
    float2 l_tile_texture_uv = float2(g_push_constants.m_tile_x, g_push_constants.m_tile_y) + l_uv * g_push_constants.m_tile_size;
    l_tile_texture_uv = frac(l_tile_texture_uv);

// Stochastic Texturing //
    // Mask UV
    float2 l_mask_uv = float2(g_push_constants.m_hex_tile_x, g_push_constants.m_hex_tile_y) + l_uv * g_push_constants.m_hex_tile_size;
    l_mask_uv = frac(l_mask_uv) * 384;

    // Mask 1 - red
    float2 l_mask_r_uv = frac(l_mask_uv);
    float l_mask_r = min(min(l_mask_r_uv.x, 1.0 - l_mask_r_uv.x), min(l_mask_r_uv.y, 1.0 - l_mask_r_uv.y));
    float3 l_mask_r_random = hash_3(floor(l_mask_uv));
    // Mask 2 - green
    float2 l_mask_g_uv = frac(l_mask_uv + 0.5);
    float l_mask_g = min(min(l_mask_g_uv.x, 1.0 - l_mask_g_uv.x), min(l_mask_g_uv.y, 1.0 - l_mask_g_uv.y));
    float3 l_mask_g_random = hash_3(floor(l_mask_uv + 0.5) - 100.0);

    // Combine masks & contrast mask
    float2 l_mask = float2(l_mask_r + 3.0, l_mask_g + 3.0);
    l_mask = pow(l_mask, 15.0);
    l_mask /= l_mask.r + l_mask.g;

    // UV Rotations
    const float l_tau = 2.0 * M_PI;
    float l_rotation_r = floor(l_mask_r_random.x * 8.0 ) / 8.0 * l_tau;
    float l_rotation_g = floor(l_mask_g_random.x * 8.0 ) / 8.0 * l_tau;
    // UV Scale if texture is diagonal
    float l_scale_r = frac(l_mask_r_random.x * 4) < 0.5 ? 1.0 : sqrt(2.0);
    float l_scale_g = frac(l_mask_g_random.x * 4) < 0.5 ? 1.0 : sqrt(2.0);
    // Transform UVs
    float2 l_new_uv_1 = rotate_2D(l_tile_texture_uv, l_rotation_r) * l_scale_r + l_mask_r_random.yz;
    float2 l_new_uv_2 = rotate_2D(l_tile_texture_uv, l_rotation_g) * l_scale_g + l_mask_g_random.yz;

    // Layers
    float2 l_splat_vt_uv = tile_uv_to_texture_uv(l_tile_uv,
                                                 g_push_constants.m_splat_tile_resolution,
                                                 g_push_constants.m_splat_tile_border);

    float4 l_layer_mask_0 = bindless_tex2d_array_sample_level(g_push_constants.m_tile_layer_mask_0_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(l_splat_vt_uv, g_push_constants.m_tile_texture_slice), 0, 0);
    float4 l_layer_mask_1 = bindless_tex2d_array_sample_level(g_push_constants.m_tile_layer_mask_1_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(l_splat_vt_uv, g_push_constants.m_tile_texture_slice), 0, 0);
    float4 l_layer_mask_2 = bindless_tex2d_array_sample_level(g_push_constants.m_tile_layer_mask_2_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(l_splat_vt_uv, g_push_constants.m_tile_texture_slice), 0, 0);
    float4 l_layer_mask_3 = bindless_tex2d_array_sample_level(g_push_constants.m_tile_layer_mask_3_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float3(l_splat_vt_uv, g_push_constants.m_tile_texture_slice), 0, 0);

    //-------------------------------------------------------------------------
    // Blending
    //-------------------------------------------------------------------------

    // Pick correct mip-level
    float l_mip_level = g_push_constants.m_mip_level;

    // Pack layers into an array
    float l_layer_mask[MAX_LAYER_COUNT];
    l_layer_mask[0]  = l_layer_mask_0.x;
    l_layer_mask[1]  = l_layer_mask_0.y;
    l_layer_mask[2]  = l_layer_mask_0.z;
    l_layer_mask[3]  = l_layer_mask_0.w;
    l_layer_mask[4]  = l_layer_mask_1.x;
    l_layer_mask[5]  = l_layer_mask_1.y;
    l_layer_mask[6]  = l_layer_mask_1.z;
    l_layer_mask[7]  = l_layer_mask_1.w;
    l_layer_mask[8]  = l_layer_mask_2.x;
    l_layer_mask[9]  = l_layer_mask_2.y;
    l_layer_mask[10] = l_layer_mask_2.z;
    l_layer_mask[11] = l_layer_mask_2.w;
    l_layer_mask[12] = l_layer_mask_3.x;
    l_layer_mask[13] = l_layer_mask_3.y;
    l_layer_mask[14] = l_layer_mask_3.z;
    l_layer_mask[15] = l_layer_mask_3.w;


// Stochastic Texturing heightmaps //

    float2 l_local_height_masked[MAX_LAYER_COUNT];
    // Pack heightmaps
    l_local_height_masked[0]  = float2(get_material(l_pbr_material_list, 0,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 0,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[1]  = float2(get_material(l_pbr_material_list, 1,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 1,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[2]  = float2(get_material(l_pbr_material_list, 2,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 2,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[3]  = float2(get_material(l_pbr_material_list, 3,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 3,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[4]  = float2(get_material(l_pbr_material_list, 4,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 4,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[5]  = float2(get_material(l_pbr_material_list, 5,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 5,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[6]  = float2(get_material(l_pbr_material_list, 6,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 6,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[7]  = float2(get_material(l_pbr_material_list, 7,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 7,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[8]  = float2(get_material(l_pbr_material_list, 8,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 8,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[9]  = float2(get_material(l_pbr_material_list, 9,  l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 9,  l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[10] = float2(get_material(l_pbr_material_list, 10, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 10, l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[11] = float2(get_material(l_pbr_material_list, 11, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 11, l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[12] = float2(get_material(l_pbr_material_list, 12, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 12, l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[13] = float2(get_material(l_pbr_material_list, 13, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 13, l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[14] = float2(get_material(l_pbr_material_list, 14, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 14, l_new_uv_2, l_mip_level).m_local_height);
    l_local_height_masked[15] = float2(get_material(l_pbr_material_list, 15, l_new_uv_1, l_mip_level).m_local_height,  get_material(l_pbr_material_list, 15, l_new_uv_2, l_mip_level).m_local_height);

    // Blend heightmaps
    float l_local_height[MAX_LAYER_COUNT];
    l_local_height[0]  = dot(l_local_height_masked[0],  l_mask);
    l_local_height[1]  = dot(l_local_height_masked[1],  l_mask);
    l_local_height[2]  = dot(l_local_height_masked[2],  l_mask);
    l_local_height[3]  = dot(l_local_height_masked[3],  l_mask);
    l_local_height[4]  = dot(l_local_height_masked[4],  l_mask);
    l_local_height[5]  = dot(l_local_height_masked[5],  l_mask);
    l_local_height[6]  = dot(l_local_height_masked[6],  l_mask);
    l_local_height[7]  = dot(l_local_height_masked[7],  l_mask);
    l_local_height[8]  = dot(l_local_height_masked[8],  l_mask);
    l_local_height[9]  = dot(l_local_height_masked[9],  l_mask);
    l_local_height[10] = dot(l_local_height_masked[10], l_mask);
    l_local_height[11] = dot(l_local_height_masked[11], l_mask);
    l_local_height[12] = dot(l_local_height_masked[12], l_mask);
    l_local_height[13] = dot(l_local_height_masked[13], l_mask);
    l_local_height[14] = dot(l_local_height_masked[14], l_mask);
    l_local_height[15] = dot(l_local_height_masked[15], l_mask);


    // Compute dominant index
    float l_max_height = l_local_height[0] * l_layer_mask[0];
    int l_dominant_index = 7;
    int l_material_count = min(g_push_constants.m_material_count, MAX_LAYER_COUNT);
    for (int i = 1; i <= l_material_count; ++i)
    {
        float height_0 = l_max_height * (1.0f - l_layer_mask[i]);
        float height_1 = l_local_height[i] * l_layer_mask[i];

        l_max_height = max(height_0, height_1);
        l_dominant_index = height_0 >= height_1 ? l_dominant_index : i;
    }


    // Read data
    quadtree_material_t l_material_r = get_material(l_pbr_material_list, l_dominant_index, l_new_uv_1, l_mip_level);
    quadtree_material_t l_material_g = get_material(l_pbr_material_list, l_dominant_index, l_new_uv_2, l_mip_level);

    // Rotate ts normal for stochastic texturing
    float l_normal_rot_mul = saturate(1.0 - l_mip_level / 5.0);   // Mipped normal maps dont point upward so when rotating them you will get lighting artifacts // devide could be changed in the future (procedural elevation has enough high frequency detail to mask most artifacts)
    l_material_r.m_normal_ts.rg = rotate_2D(l_material_r.m_normal_ts.rg - 0.5, l_rotation_r * l_normal_rot_mul) + 0.5;
    l_material_g.m_normal_ts.rg = rotate_2D(l_material_g.m_normal_ts.rg - 0.5, l_rotation_g * l_normal_rot_mul) + 0.5;

    // Linear blending
/*     l_material.m_base_color = l_material_r.m_base_color * l_mask.r + l_material_g.m_base_color * l_mask.g;
    l_material.m_normal_ts =  l_material_r.m_normal_ts  * l_mask.r + l_material_g.m_normal_ts  * l_mask.g;
    l_material.m_roughness =  dot(float2(l_material_r.m_roughness, l_material_g.m_roughness), l_mask);
    l_material.m_ao =         dot(float2(l_material_r.m_ao,        l_material_g.m_ao),        l_mask); */

    // Height Blending
    quadtree_material_t l_material;
    l_material.m_base_color = height_blend(l_material_r.m_base_color, l_material_g.m_base_color,
                                           l_local_height_masked[l_dominant_index].r, l_local_height_masked[l_dominant_index].g,
                                           l_mask);
    l_material.m_normal_ts =  height_blend(l_material_r.m_normal_ts, l_material_g.m_normal_ts,
                                           l_local_height_masked[l_dominant_index].r, l_local_height_masked[l_dominant_index].g,
                                           l_mask);
    l_material.m_roughness =  height_blend(l_material_r.m_roughness, l_material_g.m_roughness,
                                           l_local_height_masked[l_dominant_index].r, l_local_height_masked[l_dominant_index].g,
                                           l_mask);
    l_material.m_ao =         height_blend(l_material_r.m_ao, l_material_g.m_ao,
                                           l_local_height_masked[l_dominant_index].r, l_local_height_masked[l_dominant_index].g,
                                           l_mask);

    // Store material into virtual texture
    uint2 l_dst_xyz = uint2(p_dispatch_thread_id.xy);
    l_vt0_tile[l_dst_xyz] = float4(l_material.m_base_color, l_material.m_roughness);
    l_vt1_tile[l_dst_xyz] = float4(l_material.m_normal_ts, l_material.m_ao);

    //-------------------------------------------------------------------------
    // Debug output
    //-------------------------------------------------------------------------

#if OUTPUT_UV
    l_vt0_tile[l_dst_xyz] = float4(l_tile_texture_uv, 0, 0.99f);
    l_vt1_tile[l_dst_xyz] = float4(0, 0, 1.0f, 0);
#endif
}
