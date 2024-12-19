// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_population_compact_append_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------
[numthreads(POPULATION_COMPACT_APPEND_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    // Get population item and skip 'invalid' items
    uint population_item_index = dispatch_thread_id.x;

    StructuredBuffer<sb_population_item_t> population_buffer = ResourceDescriptorHeap[g_push_constants.m_population_buffer_srv];
    sb_population_item_t population_item = population_buffer[population_item_index];
    if (population_item.m_item_id == 0)
    {
        return;
    }

    RWStructuredBuffer<uint> compact_population_count = ResourceDescriptorHeap[g_push_constants.m_compact_population_count_uav];

    uint compacted_index = 0;
    InterlockedAdd(compact_population_count[0], 1, compacted_index);

    // Exit if we are exceeding buffer capacity
    if (compacted_index >= g_push_constants.m_compact_population_capacity)
    {
        uint l_original_value = 0;
        InterlockedExchange(compact_population_count[0], g_push_constants.m_compact_population_capacity, l_original_value);
        return;
    }
    
    RWStructuredBuffer<sb_population_tile_item_t> compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_uav];
    compact_population_buffer[compacted_index].m_population_item = population_item;
    compact_population_buffer[compacted_index].m_tile_index = g_push_constants.m_tile_index;
    compact_population_buffer[compacted_index].m_cell_index = population_item_index;
}
