// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_gltf_t>                  g_push_constants        : register(REGISTER_PUSH_CONSTANTS);

// UAV

// Helper functions
#include "../helper_shaders/mb_quadtree_common.hlsl"
#include "../helper_shaders/mb_wind.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
                        float4  m_position_ps   : SV_POSITION;
    nointerpolation     uint    m_instance_id   : ID0;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    nointerpolation     uint m_entity_id        : TEXCOORD0;        
#endif
};

struct ps_output_t
{
    uint2 m_ids             : SV_TARGET0;
    uint  m_classification  : SV_TARGET1;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    uint m_entity_id        : SV_TARGET2;
#endif //MB_RENDER_SELECTION_PASS_ENABLED
};

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

uint pack_classification()
{
    uint l_result = e_lighting_classification_mask_terrain;
    return l_result;
}

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ps_input_t vs_gpu_instancing(   uint p_vertex_id    : SV_VertexID,
                                uint p_instance_id  : SV_InstanceID)
{
    ps_input_t l_result = (ps_input_t)0;

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
    sb_tile_instance_t l_tile = l_tile_instances[l_render_instance.m_user_data];

    // Unpack input data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get vertex resolution
    uint l_vertex_resolution = l_quadtree_material.m_tile_size_in_vertices;

    // Get tile position from vertex id
    float2 l_tile_position = get_tile_position(l_render_item, p_vertex_id, l_vertex_resolution);

    float l_blend_mask = 0;
#if defined(TERRAIN_BLENDING)
    bool l_position_moved = false;

    // Move vertives to match neighboring tiles
    terrain_blend_mask_vertex(l_tile,
                              l_vertex_resolution,
                              l_tile_position,
                              l_position_moved,
                              l_blend_mask);

    // Get mesh data
    terrain_sample_t l_terrain_sample = sample_terrain(l_render_item, l_tile, l_tile_position, l_vertex_resolution, l_position_moved);

    // Merge vertices to match parent-tile vertex frequency
    if (l_tile.m_parent_index != TILE_NO_PARENT)
    {
        blend_with_parent(l_terrain_sample,
                          l_tile_position,
                          l_tile,
                          l_render_item,
                          l_tile_instances[l_tile.m_parent_index],
                          l_vertex_resolution,
                          l_blend_mask,
                          l_position_moved);
    }
#else
    // Get mesh data
    terrain_sample_t l_terrain_sample = sample_terrain(l_render_item, l_tile, l_tile_position, l_vertex_resolution);
#endif

#if defined(ENABLE_SKIRT)
    apply_skirt(l_tile_position,
                l_quadtree_material.m_skirt_distance_threshold_squared,
                l_quadtree_material.m_skirt_scale,
                l_render_item,
                l_tile,
                l_vertex_resolution,
                l_terrain_sample.m_position_ws_local);
#endif

    // Compute positions
    float4 l_pos_ws_local = float4(l_terrain_sample.m_position_ws_local, 1.0);
    float4 l_pos_lvs = mul(l_pos_ws_local, l_camera.m_view_local);
    float4 l_pos_ps = mul(l_pos_lvs, l_camera.m_proj);

    // Vertex shader output
    l_result.m_position_ps = l_pos_ps;
    l_result.m_instance_id = p_instance_id;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_result.m_entity_id = l_render_instance.m_entity_id;
#endif //MB_RENDER_SELECTION_PASS_ENABLED

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ps_shadow_pass(ps_input_t p_input)
{
}

//-----------------------------------------------------------------------------
void ps_main()
{
}

//-----------------------------------------------------------------------------
void ps_impostor_data_pass(ps_input_t p_input)
{
}

//-----------------------------------------------------------------------------
ps_output_t ps_visibility_pass(
    ps_input_t p_input, 
    uint p_primitive_id : SV_PrimitiveID, 
    bool p_front_face : SV_IsFrontFace)
{
    ps_output_t l_ps_output = (ps_output_t)0;

    uint l_packed_instance_id_pixel_options = pack_instance_id_pixel_options(p_input.m_instance_id, p_front_face, false, false);
    l_ps_output.m_ids = uint2(l_packed_instance_id_pixel_options, p_primitive_id);
    l_ps_output.m_classification = pack_classification();

#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_ps_output.m_entity_id = pack_entity_id(p_input.m_entity_id);
#endif

    return l_ps_output;
}

//-----------------------------------------------------------------------------
void ps_occlusion_pre_pass(ps_input_t p_input)
{

}