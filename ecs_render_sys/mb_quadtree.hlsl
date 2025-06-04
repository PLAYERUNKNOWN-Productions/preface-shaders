// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// Push constants
ConstantBuffer<cb_push_gltf_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

// Helper functions
#include "mb_lighting.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

struct ps_output_t
{
    float4 m_direct_lighting    : SV_TARGET0;
    float4 m_indirect_lighting  : SV_TARGET1;
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float2 m_velocity           : SV_TARGET2;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    uint m_entity_id            : SV_TARGET3;
#endif
};

ps_input_lighting_quadtree_t vs_gpu_instancing(     uint p_vertex_id    : SV_VertexID,
                                                    uint p_instance_id  : SV_InstanceID)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Get render item info
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    // Get material
    StructuredBuffer<sb_quadtree_material_t> l_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_quadtree_material_t l_quadtree_material = l_material_list[l_render_item.m_material_index];

    // Get patch data
    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(l_quadtree_material.m_tile_instance_buffer_index)];

    // Unpack input data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    ps_input_lighting_quadtree_t l_result = lighting_vertex_shader_quadtree(
        p_vertex_id,
        l_render_item,
        l_quadtree_material,
        l_render_instance,
        l_tile_instances,
        l_camera,
        p_instance_id);

    return l_result;
}

void ps_shadow_pass()
{
}

void ps_visibility_pass()
{
}

ps_output_t ps_main(ps_input_lighting_quadtree_t p_input)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_input.m_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Get render item info
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    // Get material
    StructuredBuffer<sb_quadtree_material_t> l_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_quadtree_material_t l_quadtree_material = l_material_list[l_render_item.m_material_index];

    // Get patch data
    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(l_quadtree_material.m_tile_instance_buffer_index)];
    sb_tile_instance_t l_tile = l_tile_instances[l_render_instance.m_user_data];

    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float4 l_direct_lighting = (float4)0;
    float4 l_indirect_lighting = (float4)0;
    lighting_pixel_shader_quadtree(
        p_input,
        l_quadtree_material,
        l_tile,
        g_push_constants.m_shadow_caster_count,
        g_push_constants.m_shadow_caster_srv,
        g_push_constants.m_gsm_srv,
        g_push_constants.m_gsm_camera_view_local_proj,
        g_push_constants.m_exposure_value,
        g_push_constants.m_dfg_texture_srv,
        g_push_constants.m_diffuse_ld_texture_srv,
        g_push_constants.m_specular_ld_texture_srv,
        g_push_constants.m_dfg_texture_size,
        g_push_constants.m_specular_ld_mip_count,
        g_push_constants.m_light_list_cbv,
        (float3x3)l_camera.m_align_ground_rotation,
        l_direct_lighting,
        l_indirect_lighting);

    ps_output_t l_ps_output;

    l_ps_output.m_direct_lighting = l_direct_lighting;
    l_ps_output.m_indirect_lighting = l_indirect_lighting;

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    l_ps_output.m_velocity = get_motion_vector_without_jitter(float2(l_camera.m_resolution_x, l_camera.m_resolution_y), p_input.m_proj_pos_curr, p_input.m_proj_pos_prev, l_camera.m_jitter, l_camera.m_jitter_prev);
#endif

#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_ps_output.m_entity_id = pack_entity_id(p_input.m_entity_id);
#endif

    return l_ps_output;
}
