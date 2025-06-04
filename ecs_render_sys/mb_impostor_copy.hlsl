// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_impostor_common.hlsl"

ConstantBuffer<cb_push_impostor_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(IMPOSTOR_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    StructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_srv];
    uint l_instance_count = l_instance_count_buffer[0];

    uint l_instance_idx = p_dispatch_thread_id.x;
    if (l_instance_idx >= l_instance_count)
    {
        return;
    }

    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    StructuredBuffer<sb_impostor_instance_t> l_instance_list = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_srv];
    StructuredBuffer<sb_impostor_item_t> l_item_list = ResourceDescriptorHeap[g_push_constants.m_item_buffer_srv];
    sb_impostor_instance_t l_instance = l_instance_list[l_instance_idx];
    sb_impostor_item_t l_item = l_item_list[l_instance.m_item_idx];

    // Culling
    if (!accept_impostor_instance(l_item, l_instance, l_camera, g_push_constants.m_hiz_map_srv, g_push_constants.m_lod_bias))
    {
        return;
    }

    // For each instance, add them to a global instance buffer sorted per item index
    StructuredBuffer<indirect_draw_impostor_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];
    uint l_command_index = l_instance.m_item_idx;

    uint l_item_count = 0;
    RWStructuredBuffer<uint> l_item_count_buffer = ResourceDescriptorHeap[g_push_constants.m_item_count_buffer_uav];
    InterlockedAdd(l_item_count_buffer[l_command_index], 1, l_item_count);

    RWStructuredBuffer<uint> l_sorted_instance_list = ResourceDescriptorHeap[g_push_constants.m_sorted_instance_buffer_uav];
    uint l_instance_base = l_command_buffer[l_command_index].m_draw.m_start_instance_location;
    l_sorted_instance_list[l_instance_base + l_item_count] = l_instance_idx;
}
