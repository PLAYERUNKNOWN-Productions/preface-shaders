// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_compact_population_update_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(TILE_POPULATION_UPDATE_THREADGROUP_SIZE, 1, 1)]
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

    StructuredBuffer<sb_population_update_t> population_update_buffer = ResourceDescriptorHeap[g_push_constants.m_population_update_buffer_srv];
    for (uint i = 0; i < g_push_constants.m_update_count; ++i)
    {
        sb_population_update_t population_update = population_update_buffer[i];

        if (population_tile_item.m_tile_index == population_update.m_tile_index &&
            population_tile_item.m_cell_index == population_update.m_cell_index)
        {
            compact_population_buffer[index].m_population_item.m_item_id = population_update.m_value;
        }
    }
}
