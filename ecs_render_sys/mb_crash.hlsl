// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_crash_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(32, 32, 1)]
void cs_main_page_fault(uint3 p_dispatch_thread_id : SV_DispatchThreadID) 
{
    // This is currently (10/12/2024) broken but not required.

    RWTexture2D<float> texture = ResourceDescriptorHeap[g_push_constants.m_invalid_uav_index];

    for(uint index = 0; index < g_push_constants.m_loop_count; ++index)
    {
        texture[p_dispatch_thread_id.xy] = 0;
    }
}

[numthreads(1024, 1, 1)]
void cs_main_device_hang(uint group_index : SV_GroupIndex)
{
    RWStructuredBuffer<int> buffer = ResourceDescriptorHeap[g_push_constants.m_buffer_uav];

    for(uint index = 0; index < g_push_constants.m_loop_count; ++index)
    {
        buffer[group_index] = buffer[group_index] + 1;
    }
}