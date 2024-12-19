// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Root constants
ConstantBuffer<cb_push_tile_population_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------
void add_instance(RWStructuredBuffer<sb_render_instance_population_t> p_instance_buffer,
                  RWStructuredBuffer<uint> p_instance_count_buffer,
                  sb_tile_instance_t p_tile_instance,
                  sb_population_item_t p_population_item,
                  uint p_id) // Temporary
{
    // Get instance index
    uint l_instance_index = 0;
    InterlockedAdd(p_instance_count_buffer[0], 1, l_instance_index);

    // Exit if we are exceeding buffer capacity
    if (l_instance_index >= g_push_constants.m_instance_buffer_capacity)
    {
        uint l_original_value = 0;
        InterlockedExchange(p_instance_count_buffer[0], g_push_constants.m_instance_buffer_capacity, l_original_value);
        return;
    }

    terrain_vertex_t l_vertex = get_terrain_vertex(p_tile_instance,
                                                   p_population_item.m_offset.xz,
                                                   g_push_constants.m_tile_height_array_index_srv,
                                                   g_push_constants.m_elevation_tile_resolution,
                                                   g_push_constants.m_elevation_tile_border);

    // Fill instance data
    sb_render_instance_population_t l_render_instance = (sb_render_instance_population_t)0;
    l_render_instance.m_render_item_idx = p_id;
    l_render_instance.m_entity_id       = p_tile_instance.m_entity_id;
    l_render_instance.m_position        = l_vertex.m_position_ws_local + p_population_item.m_offset.y * l_vertex.m_normal;
    l_render_instance.m_normal          = l_vertex.m_normal;
    l_render_instance.m_rotation        = p_population_item.m_rotation;
    l_render_instance.m_scale           = p_population_item.m_scale;

    // Add instance to buffer
    p_instance_buffer[l_instance_index] = l_render_instance;
}

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

    StructuredBuffer<sb_population_tile_item_t> l_compact_population_buffer = ResourceDescriptorHeap[g_push_constants.m_compact_population_buffer_srv];
    sb_population_tile_item_t l_population_tile_item = l_compact_population_buffer[l_population_index];
    sb_population_item_t l_population_item = l_population_tile_item.m_population_item;

    // TODO: for now we assume 0 is not items
    // Skip empty items
    if (l_population_item.m_item_id == 0)
    {
        return;
    }

    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[g_push_constants.m_tile_buffer_srv];
    sb_tile_instance_t l_tile_instance = l_tile_instances[l_population_tile_item.m_tile_index];

    // Skip tiles that are not available
    if (l_tile_instance.m_available == 0)
    {
        return;
    }

    // Reduce pressure by skipping larger tiles that won't get geometric items drawn
    if (l_tile_instance.m_tile_level < 14)
    {
        return;
    }

    StructuredBuffer<uint2> l_population_models = ResourceDescriptorHeap[g_push_constants.m_population_model_buffer_srv];
    uint l_render_item_offset = l_population_models[l_population_item.m_item_id].x;
    uint l_render_item_count = l_population_models[l_population_item.m_item_id].y;

    // Render model is empty (might be not ready or reloading)
    if (l_render_item_count == 0)
    {
        return;
    }

    // Populate instance list
    StructuredBuffer<uint> l_population_render_items = ResourceDescriptorHeap[g_push_constants.m_population_render_item_buffer_srv];
    RWStructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_uav];
    RWStructuredBuffer<sb_render_instance_population_t> l_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_uav];
    for(uint32_t l_render_item_index = l_render_item_offset; l_render_item_index < l_render_item_offset + l_render_item_count; ++l_render_item_index)
    {
        uint l_render_item_id = l_population_render_items[l_render_item_index];
        add_instance(l_instance_buffer,
                     l_instance_count_buffer,
                     l_tile_instance,
                     l_population_item,
                     l_render_item_id);
    }
}
