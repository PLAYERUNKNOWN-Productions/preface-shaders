// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_hiz_pre_pass> g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(MB_HI_Z_THREADGROUP_SIZE, 1, 1)]
void cs_main_patch_cmd_buffer(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Get command buffer
    RWStructuredBuffer<indirect_draw_instancing_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];

    // Update rootconstants
    l_command_buffer[p_dispatch_thread_id.x].m_push_constants.m_camera_cbv = g_push_constants.m_camera_cbv;
    l_command_buffer[p_dispatch_thread_id.x].m_push_constants.m_time = g_push_constants.m_time;
}

[numthreads(MB_HI_Z_THREADGROUP_SIZE, 1, 1)]
void cs_main_patch_instance_buffer(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Get instance buffer
    RWStructuredBuffer<sb_render_instance_t> l_sorted_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_final_uav];

    // Update instances
    l_sorted_instance_buffer[p_dispatch_thread_id.x].m_transform[3] += g_push_constants.m_camera_diff;
}

[numthreads(MB_HI_Z_THREADGROUP_SIZE, 1, 1)]
void cs_main_patch_tile_instance_buffer(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    RWStructuredBuffer<sb_tile_instance_t> l_tile_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_quadtree_instance_buffer_default_uav];

    // Update instances
    l_tile_instance_buffer[p_dispatch_thread_id.x].m_tile_local_to_camera_local += g_push_constants.m_camera_diff;
}