// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_tile_impostor_population_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------
[numthreads(TILE_POPULATION_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    StructuredBuffer<uint> l_compact_population_count_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_count_buffer_srv];
    uint l_population_index = p_dispatch_thread_id.x;
    if (l_population_index >= l_compact_population_count_buffer[0])
    {
        return;
    }

    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[g_push_constants.m_tile_buffer_srv];
    StructuredBuffer<sb_population_tile_item_t> l_compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_buffer_srv];
    sb_population_tile_item_t l_population_tile_item = l_compact_population_buffer[l_population_index];
    sb_tile_instance_t l_tile_instance = l_tile_instances[l_population_tile_item.m_tile_index];

    // Skip tiles that are not available
    if (l_tile_instance.m_available == 0)
    {
        return;
    }

    // Get instance index
    RWStructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_uav];
    uint l_instance_index = 0;
    InterlockedAdd(l_instance_count_buffer[0], 1, l_instance_index);

    // Exit if we are exceeding buffer capacity
    if (l_instance_index >= g_push_constants.m_instance_buffer_capacity)
    {
        uint l_original_value = 0;
        InterlockedExchange(l_instance_count_buffer[0], g_push_constants.m_instance_buffer_capacity, l_original_value);
        return;
    }

    sb_population_item_t l_population_item = l_population_tile_item.m_population_item;
    terrain_vertex_t l_vertex = get_terrain_vertex(l_tile_instance,
                                                   l_population_item.m_offset.xz,
                                                   g_push_constants.m_tile_height_array_index_srv,
                                                   g_push_constants.m_elevation_tile_resolution,
                                                   g_push_constants.m_elevation_tile_border);

    // Fill instance data
    sb_impostor_instance_t l_instance = (sb_impostor_instance_t)0;
    l_instance.m_position  = l_vertex.m_position_ws_local + l_population_item.m_offset.y * l_vertex.m_normal;
    l_instance.m_up_vector = l_vertex.m_normal;
    l_instance.m_angle     = l_population_item.m_rotation;
    l_instance.m_item_idx  = l_population_item.m_item_id;
    l_instance.m_scale     = l_population_item.m_scale;

    RWStructuredBuffer<sb_impostor_instance_t> l_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_uav];
    l_instance_buffer[l_instance_index] = l_instance;
}
