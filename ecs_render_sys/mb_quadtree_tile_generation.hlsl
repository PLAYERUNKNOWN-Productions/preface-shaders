// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_tile_generation_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------

[numthreads(TILE_GENERATION_THREADGROUP_SIZE, TILE_GENERATION_THREADGROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Get inputs
    StructuredBuffer<float> l_elevation_buffer = ResourceDescriptorHeap[g_push_constants.m_elevation_buffer_srv];
    StructuredBuffer<float> l_splat_buffer     = ResourceDescriptorHeap[g_push_constants.m_splat_buffer_srv];

    // Get output textures
    RWTexture2DArray<float> l_height_map       = ResourceDescriptorHeap[g_push_constants.m_tile_heightmap_index];
    RWTexture2DArray<float4> l_texture_map     = ResourceDescriptorHeap[g_push_constants.m_tile_texturemap_index];
    RWTexture2DArray<float4> l_layer_mask_0    = ResourceDescriptorHeap[g_push_constants.m_tile_layer_mask_0_index];
    RWTexture2DArray<float4> l_layer_mask_1    = ResourceDescriptorHeap[g_push_constants.m_tile_layer_mask_1_index];
    RWTexture2DArray<float4> l_layer_mask_2    = ResourceDescriptorHeap[g_push_constants.m_tile_layer_mask_2_index];
    RWTexture2DArray<float4> l_layer_mask_3    = ResourceDescriptorHeap[g_push_constants.m_tile_layer_mask_3_index];

    // Store the result
    if (all(p_dispatch_thread_id.xy < g_push_constants.m_elevation_resolution))
    {
        uint3 l_dst_xyz = uint3(p_dispatch_thread_id.xy, g_push_constants.m_tile_texture_slice);
        uint l_elevation_src_index = p_dispatch_thread_id.x + p_dispatch_thread_id.y * g_push_constants.m_elevation_resolution.x;
        l_height_map[l_dst_xyz]     = l_elevation_buffer[g_push_constants.m_elevation_data_offset + l_elevation_src_index];
    }

    if (all(p_dispatch_thread_id.xy < g_push_constants.m_texture_resolution))
    {
        uint3 l_dst_xyz = uint3(p_dispatch_thread_id.xy, g_push_constants.m_tile_texture_slice);
        uint l_texture_src_index = p_dispatch_thread_id.x + p_dispatch_thread_id.y * g_push_constants.m_texture_resolution.x;

        if(g_push_constants.m_texture_buffer_srv != RAL_NULL_BINDLESS_INDEX)
        {
            StructuredBuffer<float> l_texture_buffer = ResourceDescriptorHeap[g_push_constants.m_texture_buffer_srv];

            //TODO, instead of always loading channel 0, 1, 2 we should load the channels selected by the user
            float r = l_texture_buffer[g_push_constants.m_texture_data_offset + l_texture_src_index + 0 * g_push_constants.m_texture_channel_offset];
            float g = l_texture_buffer[g_push_constants.m_texture_data_offset + l_texture_src_index + 1 * g_push_constants.m_texture_channel_offset];
            float b = l_texture_buffer[g_push_constants.m_texture_data_offset + l_texture_src_index + 2 * g_push_constants.m_texture_channel_offset];

            l_texture_map[l_dst_xyz] = float4(r, g, b, 0);
        }
    }

    // Splatmap
    if (all(p_dispatch_thread_id.xy < g_push_constants.m_splat_resolution))
    {
        uint3 l_dst_xyz = uint3(p_dispatch_thread_id.xy, g_push_constants.m_tile_texture_slice);
        uint l_splat_src_index = p_dispatch_thread_id.x + p_dispatch_thread_id.y * g_push_constants.m_splat_resolution.x;
        if (g_push_constants.m_splat_channels == 0xFFFFFFFF)
        {
            l_layer_mask_0[l_dst_xyz] = float4(1, 0, 0, 0);
            l_layer_mask_1[l_dst_xyz] = (float4)0;
            l_layer_mask_2[l_dst_xyz] = (float4)0;
            l_layer_mask_3[l_dst_xyz] = (float4)0;
        }
        else
        {
            l_layer_mask_0[l_dst_xyz].x = g_push_constants.m_splat_channels > 0  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 0  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_0[l_dst_xyz].y = g_push_constants.m_splat_channels > 1  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 1  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_0[l_dst_xyz].z = g_push_constants.m_splat_channels > 2  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 2  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_0[l_dst_xyz].w = g_push_constants.m_splat_channels > 3  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 3  * g_push_constants.m_splat_channel_offset] : 0;

            l_layer_mask_1[l_dst_xyz].x = g_push_constants.m_splat_channels > 4  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 4  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_1[l_dst_xyz].y = g_push_constants.m_splat_channels > 5  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 5  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_1[l_dst_xyz].z = g_push_constants.m_splat_channels > 6  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 6  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_1[l_dst_xyz].w = g_push_constants.m_splat_channels > 7  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 7  * g_push_constants.m_splat_channel_offset] : 0;

            l_layer_mask_2[l_dst_xyz].x = g_push_constants.m_splat_channels > 8  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 8  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_2[l_dst_xyz].y = g_push_constants.m_splat_channels > 9  ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 9  * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_2[l_dst_xyz].z = g_push_constants.m_splat_channels > 10 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 10 * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_2[l_dst_xyz].w = g_push_constants.m_splat_channels > 11 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 11 * g_push_constants.m_splat_channel_offset] : 0;

            l_layer_mask_3[l_dst_xyz].x = g_push_constants.m_splat_channels > 12 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 12 * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_3[l_dst_xyz].y = g_push_constants.m_splat_channels > 13 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 13 * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_3[l_dst_xyz].z = g_push_constants.m_splat_channels > 14 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 14 * g_push_constants.m_splat_channel_offset] : 0;
            l_layer_mask_3[l_dst_xyz].w = g_push_constants.m_splat_channels > 15 ? l_splat_buffer[g_push_constants.m_splat_data_offset + l_splat_src_index + 15 * g_push_constants.m_splat_channel_offset] : 0;
        }
    }
}
