// Copyright:   PlayerUnknown Productions BV

// #define OCCLUSION_CULL_DEBUG_DRAW

#include "../helper_shaders/mb_common.hlsl"
#include "mb_instancing_common.hlsl"

// Push constants
ConstantBuffer<cb_push_instancing_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(INSTANCING_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Get instance count
    StructuredBuffer<uint> l_instance_count_buffer = ResourceDescriptorHeap[g_push_constants.m_instance_count_buffer_srv];
    uint l_instance_count = l_instance_count_buffer[0];

    // Skip instances that are outside of bounds
    if (p_dispatch_thread_id.x >= l_instance_count)
    {
        return;
    }

    uint l_instance_index = p_dispatch_thread_id.x;

    // Get camera
    ConstantBuffer<cb_camera_t> l_camera      = ResourceDescriptorHeap[g_push_constants.m_push_constants_gltf.m_camera_cbv];
    ConstantBuffer<cb_camera_t> l_lod_camera  = ResourceDescriptorHeap[g_push_constants.m_lod_camera_cbv];

    // Get render instance
    sb_render_instance_t l_render_instance = get_instance_with_conversion(g_push_constants.m_instance_buffer_srv, l_instance_index);

    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_push_constants_gltf.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    float3 l_lod_camera_offset = float3(g_push_constants.m_lod_camera_offset_x, g_push_constants.m_lod_camera_offset_y, g_push_constants.m_lod_camera_offset_z);

#if defined(MB_READ_RENDERED_INSTANCES)
    StructuredBuffer<uint> l_rendered_instances_buffer = ResourceDescriptorHeap[g_push_constants.m_rendered_instances_buffer_srv];
    bool l_already_rendered = l_rendered_instances_buffer[l_instance_index] == 1;
    if(l_already_rendered)
    {
        return;
    }
#endif

    float4x3 l_transform =
#if defined(MB_USE_LAST_FRAME_MATRICES)
        l_render_instance.m_transform_prev;
#else
        l_render_instance.m_transform;
#endif

    bool l_is_terrain = l_render_instance.m_render_item_idx == 0;
    if (!accept_render_item(l_render_item,
                            l_transform,
                            l_render_instance.m_custom_culling_scale,
                            l_camera,
                            l_lod_camera,
                            l_lod_camera_offset,
                            g_push_constants.m_hiz_map_srv,
                            g_push_constants.m_lod_bias,
                            g_push_constants.m_forced_lod,
                            l_is_terrain))
    {
        return;
    }

#if defined(MB_WRITE_RENDERED_INSTANCES)
#   error counting pass should never write this data!
#endif

    // Get command
    RWStructuredBuffer<indirect_draw_instancing_t> l_command_buffer = ResourceDescriptorHeap[g_push_constants.m_command_buffer_uav];

    // Increment render item counter
    uint l_command_index = l_render_instance.m_render_item_idx;
    InterlockedAdd(l_command_buffer[l_command_index].m_draw.m_instance_count, 1);
}
