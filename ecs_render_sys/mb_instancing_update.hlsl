// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// Push constants
ConstantBuffer<cb_push_instancing_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(INSTANCING_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip items that are outside of bounds
    if (p_dispatch_thread_id.x >= g_push_constants.m_render_item_count)
    {
        return;
    }

    // Get command
    RWStructuredBuffer<indirect_draw_instancing_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];
    RWStructuredBuffer<uint64_t> l_command_predication_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_predication_uav];

    // Update offsets
    uint l_offset = g_push_constants.m_push_constants_gltf.m_render_instance_buffer_offset;

    for (uint l_render_item_index = 0; l_render_item_index < p_dispatch_thread_id.x; ++l_render_item_index)
    {
        // Increment offset by amount of instances
        l_offset += l_command_buffer[l_render_item_index].m_draw.m_instance_count;
    }

    // Fill command buffer
    l_command_buffer[p_dispatch_thread_id.x].m_draw.m_start_instance_location                  = l_offset;
    l_command_buffer[p_dispatch_thread_id.x].m_push_constants.m_render_instance_buffer_offset  = l_offset;

    if(l_command_buffer[p_dispatch_thread_id.x].m_draw.m_instance_count != 0)
    {
        l_command_predication_buffer[p_dispatch_thread_id.x] = 1;
    }
}
