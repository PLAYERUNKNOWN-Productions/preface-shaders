// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_instancing_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(INSTANCING_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip items that are outside of bounds
    if (p_dispatch_thread_id.x >= g_push_constants.m_render_item_count)
    {
        return;
    }

    uint l_command_index = p_dispatch_thread_id.x;

    // Get render item
    uint l_render_item_index = l_command_index; // 1 command per 1 item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_item_index];

    // Get command buffer
    RWStructuredBuffer<indirect_draw_instancing_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];

    // Get scratch buffer
    RWStructuredBuffer<uint> l_scratch_buffer = ResourceDescriptorHeap[g_push_constants.m_scratch_buffer_index];

    // Reset scratch buffer
    l_scratch_buffer[l_command_index] = 0;

    // Reset commands
    l_command_buffer[l_command_index].m_draw.m_index_count_per_instance = l_render_item.m_index_count;
    l_command_buffer[l_command_index].m_draw.m_instance_count           = 0;
    l_command_buffer[l_command_index].m_draw.m_start_index_location     = 0;
    l_command_buffer[l_command_index].m_draw.m_base_vertex_location     = 0;
    l_command_buffer[l_command_index].m_draw.m_start_instance_location  = 0;

    l_command_buffer[l_command_index].m_push_constants                  = g_push_constants.m_push_constants_gltf;
}
