// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "mb_instancing_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_instancing_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(INSTANCING_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Get instance count
    StructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_srv];
    uint l_instance_count = l_instance_count_buffer[0];

    // Skip instances that are outside of bounds
    if (p_dispatch_thread_id.x >= l_instance_count)
    {
        return;
    }

    // Get camera
    ConstantBuffer<cb_camera_t> l_camera      = ResourceDescriptorHeap[g_push_constants.m_push_constants_gltf.m_camera_cbv];
    ConstantBuffer<cb_camera_t> l_lod_camera  = ResourceDescriptorHeap[g_push_constants.m_lod_camera_cbv];

    // Get render instance
    uint l_instance_index = p_dispatch_thread_id.x;
    sb_render_instance_t l_render_instance = get_instance_with_conversion(g_push_constants.m_instance_buffer_srv, l_instance_index);

    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    float3 l_lod_camera_offset = float3(g_push_constants.m_lod_camera_offset_x, g_push_constants.m_lod_camera_offset_y, g_push_constants.m_lod_camera_offset_z);
    if (!accept_render_item(l_render_item, 
                            l_render_instance, 
                            l_camera, 
                            l_lod_camera, 
                            l_lod_camera_offset, 
                            g_push_constants.m_impostor_distance, 
                            g_push_constants.m_hiz_map_srv, 
                            g_push_constants.m_lod_bias))
    {
        return;
    }

    // World transform
    float4x3 l_world_transform = mul_world_matricies(l_render_item.m_transform, l_render_instance.m_transform);
    float4x3 l_world_transform_prev = mul_world_matricies(l_render_item.m_transform, l_render_instance.m_transform_prev);

    //
    RWStructuredBuffer<sb_render_instance_t> l_render_instance_buffer_final = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_final_index];

    // Get command
    StructuredBuffer<indirect_draw_instancing_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_srv];

    // Get scratch buffer
    RWStructuredBuffer<uint> l_scratch_buffer = ResourceDescriptorHeap[g_push_constants.m_scratch_buffer_index];

    // Get scratch buffer
    uint l_command_index = l_render_instance.m_render_item_idx;
    uint l_offset = l_command_buffer[l_command_index].m_draw.m_start_instance_location;

    // Increment render item counter
    uint l_instance_index_final = 0;
    InterlockedAdd(l_scratch_buffer[l_command_index], 1, l_instance_index_final);

    //! \todo No copy is needed - we can read from the buffer using index. Profile to compare if it is faster than without indirection
    l_render_instance_buffer_final[l_instance_index_final + l_offset] = l_render_instance;
    l_render_instance_buffer_final[l_instance_index_final + l_offset].m_transform = l_world_transform;
    l_render_instance_buffer_final[l_instance_index_final + l_offset].m_transform_prev = l_world_transform_prev;
}
