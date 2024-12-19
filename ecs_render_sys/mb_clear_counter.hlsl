// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_clear_counter_t>     g_push_constants        : register(REGISTER_PUSH_CONSTANTS);

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
    RWStructuredBuffer<uint> l_counter = ResourceDescriptorHeap[g_push_constants.m_counter_resource_uav];
    l_counter[0] = 0;
}


