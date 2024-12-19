// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_population_compact_reset_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------
[numthreads(POPULATION_COMPACT_RESET_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    StructuredBuffer<uint> compact_population_count_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_count_srv];
    uint compact_population_count = compact_population_count_buffer[0];

    uint index = dispatch_thread_id.x;
    if (index >= compact_population_count)
    {
        return;
    }

    RWStructuredBuffer<sb_population_tile_item_t> compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_uav];
    sb_population_tile_item_t population_tile_item = compact_population_buffer[index];

    if (population_tile_item.m_tile_index == g_push_constants.m_tile_index)
    {
        compact_population_buffer[index].m_tile_index = 0xFFFFFFFF;
        compact_population_buffer[index].m_population_item.m_item_id = 0;
    }
}
