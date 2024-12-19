// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_debug_rendering_indirect_t>      g_push_constants            : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(1, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    RWStructuredBuffer<indirect_draw_debug_rendering_t> l_draw_command_buffer = ResourceDescriptorHeap[g_push_constants.m_draw_command_buffer_uav];
    StructuredBuffer<uint> l_counter = ResourceDescriptorHeap[g_push_constants.m_counter_resource_srv];

#ifdef TRI
    l_draw_command_buffer[0].m_push_constants.m_view_proj_matrix = g_push_constants.m_view_proj_matrix;
    l_draw_command_buffer[0].m_push_constants.m_vertex_buffer_srv = g_push_constants.m_vertex_buffer_srv;
    l_draw_command_buffer[0].m_draw.m_vertex_count_per_instance = l_counter[0];
    l_draw_command_buffer[0].m_draw.m_instance_count = 1;
    l_draw_command_buffer[0].m_draw.m_start_vertex_location = 0;
    l_draw_command_buffer[0].m_draw.m_start_instance_location = 0;
#elif defined(LINE)
    l_draw_command_buffer[1].m_push_constants.m_view_proj_matrix = g_push_constants.m_view_proj_matrix;
    l_draw_command_buffer[1].m_push_constants.m_vertex_buffer_srv = g_push_constants.m_vertex_buffer_srv;
    l_draw_command_buffer[1].m_draw.m_vertex_count_per_instance = l_counter[0];
    l_draw_command_buffer[1].m_draw.m_instance_count = 1;
    l_draw_command_buffer[1].m_draw.m_start_vertex_location = 0;
    l_draw_command_buffer[1].m_draw.m_start_instance_location = 0;
#endif
}


