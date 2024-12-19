// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_tile_population_t> g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------

[numthreads(1, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Check bounds
    if (p_dispatch_thread_id.x >= 1)
    {
        return;
    }

    // Clear the instance counter
    RWStructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_uav];
    l_instance_count_buffer[0] = 0;
}
