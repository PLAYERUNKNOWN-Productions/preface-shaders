// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#define MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE

#define MB_COMPUTE

#include "../helper_shaders/mb_quadtree_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_deferred_lighting_t>     g_push_constants        : register(REGISTER_PUSH_CONSTANTS);

#define GENERATE_TBN 1
#include "mb_lighting.hlsl"

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

//#define MORTON_SWIZZLE

//ideally we should do 8x8 for amd but nvidia only supports 32 thread per wave
[numthreads(DEFERRED_LIGHTING_THREAD_GROUP_SIZE_X, DEFERRED_LIGHTING_THREAD_GROUP_SIZE_Y, 1)]
#if defined(MORTON_SWIZZLE)
void cs_main(uint p_group_index : SV_GroupIndex, uint3 p_group_id : SV_GroupID)
#else
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
#endif
{
#if defined(MORTON_SWIZZLE)
    uint2 l_group_thread_id = remap_lane_8x8(p_group_index);
    uint3 l_pixel = uint3(p_group_id.xy * 8 + l_group_thread_id, 0);
#else
    uint3 l_pixel = p_dispatch_thread_id;
#endif
    Texture2D<uint> l_visibility_classification_texture = ResourceDescriptorHeap[g_push_constants.m_lighting_classification_texture_srv];
    uint l_pixel_classification_mask = l_visibility_classification_texture.Load(l_pixel);
    uint l_pass_classification_mask = get_classification_mask_from_type((lighting_classification_types_t)g_push_constants.m_lighting_classification_type);

    if(l_pixel_classification_mask == 0 || l_pixel_classification_mask != l_pass_classification_mask)
    {
        return;
    }
    
    RWTexture2D<float4> l_direct_lighting_rt                = ResourceDescriptorHeap[g_push_constants.m_direct_lighting_uav];
    RWTexture2D<float4> l_indirect_lighting_rt              = ResourceDescriptorHeap[g_push_constants.m_indirect_lighting_uav];
    Texture2D<uint2> l_visibility_buffer                    = ResourceDescriptorHeap[g_push_constants.m_visibility_buffer_srv];
    ConstantBuffer<cb_camera_t> l_camera                    = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    uint2 l_ids = l_visibility_buffer.Load(l_pixel);
    uint l_tri_id = l_ids.y;
    uint l_instance_id = 0;
    bool l_front_face = false;
    bool l_wind = false;
    bool l_wind_small = false;
    unpack_instance_id_pixel_options(l_ids.x, l_instance_id, l_front_face, l_wind, l_wind_small);	

    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[l_instance_id];

    // Get render item info
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    float2 l_uv_pixel = (float2(l_pixel.xy) + .5f) / g_push_constants.m_dst_resolution;

    // Get material
    StructuredBuffer<sb_quadtree_material_t> l_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_quadtree_material_t l_quadtree_material = l_material_list[l_render_item.m_material_index];

    StructuredBuffer<sb_tile_instance_t> l_tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(l_quadtree_material.m_tile_instance_buffer_index)];

    ps_input_lighting_quadtree_t l_vertex_shader_result[3];

    [unroll]
    for(uint l_i = 0; l_i < 3; ++l_i)
    {
        uint l_vertex_id = (l_tri_id * 3) + l_i;
        uint l_index = get_vertex_mesh_index(l_vertex_id, l_render_item);

        l_vertex_shader_result[l_i] = lighting_vertex_shader_quadtree(
            l_index,
            l_render_item,
            l_quadtree_material,
            l_render_instance,
            l_tile_instances,
            l_camera,
            l_instance_id);
    }

    //calculate derivatives
    float2 l_pixel_ndc = (l_uv_pixel * 2.0f) - 1.0f;
    l_pixel_ndc.y = -l_pixel_ndc.y;
    barycentric_deriv_t l_full_deriv = calc_full_bary(
        l_vertex_shader_result[0].m_position_ps,
        l_vertex_shader_result[1].m_position_ps,
        l_vertex_shader_result[2].m_position_ps,
        l_pixel_ndc,
        g_push_constants.m_dst_resolution);

    float3 l_position_local_x_deriv     = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_local.x, l_vertex_shader_result[1].m_position_local.x, l_vertex_shader_result[2].m_position_local.x);
    float3 l_position_local_y_deriv     = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_local.y, l_vertex_shader_result[1].m_position_local.y, l_vertex_shader_result[2].m_position_local.y);

    float3 l_position_ws_local_x_deriv  = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.x, l_vertex_shader_result[1].m_position_ws_local.x, l_vertex_shader_result[2].m_position_ws_local.x);
    float3 l_position_ws_local_y_deriv  = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.y, l_vertex_shader_result[1].m_position_ws_local.y, l_vertex_shader_result[2].m_position_ws_local.y);
    float3 l_position_ws_local_z_deriv  = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.z, l_vertex_shader_result[1].m_position_ws_local.z, l_vertex_shader_result[2].m_position_ws_local.z);
    float3 l_planet_normal_ws           = interpolate(l_full_deriv, l_vertex_shader_result[0].m_planet_normal_ws, l_vertex_shader_result[1].m_planet_normal_ws, l_vertex_shader_result[2].m_planet_normal_ws);
    float3 l_surface_normal_ws          = interpolate(l_full_deriv, l_vertex_shader_result[0].m_surface_normal_ws, l_vertex_shader_result[1].m_surface_normal_ws, l_vertex_shader_result[2].m_surface_normal_ws);
    float  l_blend_mask                 = interpolate(l_full_deriv, l_vertex_shader_result[0].m_blend_mask, l_vertex_shader_result[1].m_blend_mask, l_vertex_shader_result[2].m_blend_mask);

    float2 l_position_local         = float2(l_position_local_x_deriv[0], l_position_local_y_deriv[0]);
    float2 l_position_local_ddx     = float2(l_position_local_x_deriv[1], l_position_local_y_deriv[1]);
    float2 l_position_local_ddy     = float2(l_position_local_x_deriv[2], l_position_local_y_deriv[2]);
    float3 l_position_ws_local          = float3(l_position_ws_local_x_deriv[0], l_position_ws_local_y_deriv[0], l_position_ws_local_z_deriv[0]);
    float3 l_position_ws_local_ddx      = float3(l_position_ws_local_x_deriv[1], l_position_ws_local_y_deriv[1], l_position_ws_local_z_deriv[1]);
    float3 l_position_ws_local_ddy      = float3(l_position_ws_local_x_deriv[2], l_position_ws_local_y_deriv[2], l_position_ws_local_z_deriv[2]);

    ps_input_lighting_quadtree_t l_ps_lighting_input = (ps_input_lighting_quadtree_t)0;
    l_ps_lighting_input.m_position_ps = (float4)0; //unused in the vbuffering version of quadtree shading
    l_ps_lighting_input.m_position_local = l_position_local;
    l_ps_lighting_input.m_planet_normal_ws = l_planet_normal_ws;
    l_ps_lighting_input.m_surface_normal_ws = l_surface_normal_ws;
    l_ps_lighting_input.m_position_ws_local = l_position_ws_local;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_ps_lighting_input.m_entity_id = l_vertex_shader_result[0].m_entity_id;
#endif
    l_ps_lighting_input.m_instance_id = l_instance_id;
    l_ps_lighting_input.m_blend_mask = l_blend_mask;


    //! /todo pass buffers to functions instead of loaded data
    sb_tile_instance_t l_tile = l_tile_instances[l_render_instance.m_user_data];

    float4 l_direct_lighting = (float4)0;
    float4 l_indirect_lighting = (float4)0;
    lighting_pixel_shader_quadtree(
        l_ps_lighting_input,
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
        l_position_ws_local_ddx,
        l_position_ws_local_ddy,
        l_position_local_ddx,
        l_position_local_ddy,
        l_direct_lighting,
        l_indirect_lighting);

    l_direct_lighting_rt[l_pixel.xy] = l_direct_lighting;
    l_indirect_lighting_rt[l_pixel.xy] = l_indirect_lighting;
}
