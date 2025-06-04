// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_impostor_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(IMPOSTOR_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    uint l_item_idx = p_dispatch_thread_id.x;
    if (l_item_idx >= g_push_constants.m_item_count)
    {
        return;
    }

    // Reset item counts, used later
    RWStructuredBuffer<uint> l_item_count_buffer = ResourceDescriptorHeap[g_push_constants.m_item_count_buffer_uav];
    l_item_count_buffer[l_item_idx] = 0;

    // Reset commands
    RWStructuredBuffer<indirect_draw_impostor_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];
    l_command_buffer[l_item_idx].m_draw.m_vertex_count_per_instance = 6; // One quad per impostor
    l_command_buffer[l_item_idx].m_draw.m_instance_count            = 0;
    l_command_buffer[l_item_idx].m_draw.m_start_vertex_location     = 0;
    l_command_buffer[l_item_idx].m_draw.m_start_instance_location   = 0;

    // Initialize command constants (will patch in a latter pass)
    l_command_buffer[l_item_idx].m_push_constants = g_push_constants;
}
