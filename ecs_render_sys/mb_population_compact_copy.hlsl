// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_population_compact_copy_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------
[numthreads(POPULATION_COMPACT_COPY_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    StructuredBuffer<uint> src_compact_population_count_buffer = ResourceDescriptorHeap[g_push_constants.m_src_compact_population_count_srv];
    uint src_compact_population_count = src_compact_population_count_buffer[0];

    uint src_index = p_dispatch_thread_id.x;
    if (src_index >= src_compact_population_count)
    {
        return;
    }

    StructuredBuffer<sb_population_tile_item_t> src_compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_src_compact_population_srv];
    sb_population_tile_item_t src_population_tile_item = src_compact_population_buffer[src_index];
    if (src_population_tile_item.m_population_item.m_item_id == 0)
    {
        return;
    }

    RWStructuredBuffer<uint> dst_compact_population_count_buffer = ResourceDescriptorHeap[g_push_constants.m_dst_compact_population_count_uav];
    RWStructuredBuffer<sb_population_tile_item_t> dst_compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_dst_compact_population_uav];

    // No need to check for capacity, the copy can't never go over capacity
    uint dst_index = 0;
    InterlockedAdd(dst_compact_population_count_buffer[0], 1, dst_index);

    dst_compact_population_buffer[dst_index] = src_population_tile_item;
}
