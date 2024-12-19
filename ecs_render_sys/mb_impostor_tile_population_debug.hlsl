#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------
ConstantBuffer<cb_push_tile_impostor_population_debug_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Compute shader
//-----------------------------------------------------------------------------
[numthreads(1, 1, 1)]
void cs_main()
{
    // Get instance index, and exit if we are exceeding buffer capacity
    RWStructuredBuffer<uint> instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_uav];
    uint instance_index = 0;
    InterlockedAdd(instance_count_buffer[0], 1, instance_index);

    if (instance_index >= g_push_constants.m_instance_buffer_capacity)
    {
        uint original_value = 0;
        InterlockedExchange(instance_count_buffer[0], g_push_constants.m_instance_buffer_capacity, original_value);
        return;
    }

    // Fill instance data
    sb_impostor_instance_t instance = (sb_impostor_instance_t)0;
    instance.m_position  = g_push_constants.m_position;
    instance.m_up_vector = g_push_constants.m_up_vector;
    instance.m_angle     = g_push_constants.m_angle;
    instance.m_item_idx  = g_push_constants.m_item_idx;
    instance.m_scale     = g_push_constants.m_scale;

    RWStructuredBuffer<sb_impostor_instance_t> instance_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_uav];
    instance_buffer[instance_index] = instance;
}
