// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_impostor_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------
[numthreads(1, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (p_dispatch_thread_id.x >= 1)
    {
        return;
    }

    RWStructuredBuffer<indirect_draw_impostor_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];

    uint l_offset = 0;
    for (uint l_item_idx = 0; l_item_idx < g_push_constants.m_item_count; ++l_item_idx)
    {
        l_command_buffer[l_item_idx].m_draw.m_start_instance_location           = l_offset;
        l_command_buffer[l_item_idx].m_push_constants.m_start_instance_location = l_offset;

        l_offset += l_command_buffer[l_item_idx].m_draw.m_instance_count;
    }
}
