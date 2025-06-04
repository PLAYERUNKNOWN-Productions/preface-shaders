// Copyright:   PlayerUnknown Productions BV

#define MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
#define MB_VIRTUAL_TEXTURE_WRITE_FEEDBACK

// This is a temporary flag than makes ALL texture perform dynamic check if those are virtual or not
// It has a low performance and should be removed, when visibility buffer will have a proper material support
#define MB_TEST_ALL_PBR_TEXTURES_FOR_TILED_RESOURCES

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_deferred_lighting_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#define MB_COMPUTE
#define GENERATE_TBN 1
#define MB_SCALARIZE_PBR_MATERIAL_SAMPLING
//#define MORTON_SWIZZLE

#include "mb_lighting.hlsl"

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
    RWTexture2D<float2> l_velocity_rt                       = ResourceDescriptorHeap[g_push_constants.m_velocity_uav];
    Texture2D<uint2> l_visibility_buffer                    = ResourceDescriptorHeap[g_push_constants.m_visibility_buffer_srv];
    ConstantBuffer<cb_camera_t> l_camera                    = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    uint2 l_ids = l_visibility_buffer.Load(l_pixel);
    uint l_tri_id = l_ids.y;
    uint l_instance_id = 0;
    bool l_front_face = false;
    bool l_wind = false;
    bool l_wind_small = false;
    unpack_instance_id_pixel_options(l_ids.x, l_instance_id, l_front_face, l_wind, l_wind_small);

    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[l_instance_id];

    StructuredBuffer<sb_render_item_t> l_render_items = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items[l_render_instance.m_render_item_idx];

    ps_input_lighting_t l_vertex_shader_result[3];

    [unroll]
    for(uint l_i = 0; l_i < 3; ++l_i)
    {
        uint l_vertex_id = (l_tri_id * 3) + l_i;
        uint l_index = get_vertex_mesh_index(l_vertex_id, l_render_item);

        l_vertex_shader_result[l_i] = lighting_vertex_shader(
            l_index,
            l_render_item,
            l_render_instance,
            l_wind,
            l_wind_small,
            g_push_constants.m_time,
            g_push_constants.m_time_prev,
            l_camera,
            l_instance_id);
    }

    float2 l_uv_pixel = (float2(l_pixel.xy) + .5f) / g_push_constants.m_dst_resolution;

    //calculate derivatives
    float2 l_pixel_ndc = (l_uv_pixel * 2.0f) - 1.0f;
    l_pixel_ndc.y = -l_pixel_ndc.y;
    barycentric_deriv_t l_full_deriv = calc_full_bary(
        l_vertex_shader_result[0].m_position_ps,
        l_vertex_shader_result[1].m_position_ps,
        l_vertex_shader_result[2].m_position_ps,
        l_pixel_ndc,
        g_push_constants.m_dst_resolution,
        l_camera.m_render_scale);

    float3 l_uv_x_deriv     = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_texcoord0.x, l_vertex_shader_result[1].m_texcoord0.x, l_vertex_shader_result[2].m_texcoord0.x);
    float3 l_uv_y_deriv     = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_texcoord0.y, l_vertex_shader_result[1].m_texcoord0.y, l_vertex_shader_result[2].m_texcoord0.y);
    float3 l_pos_ls_x_deriv = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.x, l_vertex_shader_result[1].m_position_ws_local.x, l_vertex_shader_result[2].m_position_ws_local.x);
    float3 l_pos_ls_y_deriv = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.y, l_vertex_shader_result[1].m_position_ws_local.y, l_vertex_shader_result[2].m_position_ws_local.y);
    float3 l_pos_ls_z_deriv = interpolate_with_deriv(l_full_deriv, l_vertex_shader_result[0].m_position_ws_local.z, l_vertex_shader_result[1].m_position_ws_local.z, l_vertex_shader_result[2].m_position_ws_local.z);

    float4 l_pos_ps         = interpolate(l_full_deriv, l_vertex_shader_result[0].m_position_ps, l_vertex_shader_result[1].m_position_ps, l_vertex_shader_result[2].m_position_ps);
    float3 l_normal         = interpolate(l_full_deriv, l_vertex_shader_result[0].m_normal, l_vertex_shader_result[1].m_normal, l_vertex_shader_result[2].m_normal);
#if (GENERATE_TBN == 0)
    float3 l_tangent        = interpolate(l_full_deriv, l_vertex_shader_result[0].m_tangent, l_vertex_shader_result[1].m_tangent, l_vertex_shader_result[2].m_tangent);
    float3 l_binormal       = interpolate(l_full_deriv, l_vertex_shader_result[0].m_binormal, l_vertex_shader_result[1].m_binormal, l_vertex_shader_result[2].m_binormal);
#endif
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float3 l_proj_pos_curr  = interpolate(l_full_deriv, l_vertex_shader_result[0].m_proj_pos_curr, l_vertex_shader_result[1].m_proj_pos_curr, l_vertex_shader_result[2].m_proj_pos_curr);
    float3 l_proj_pos_prev  = interpolate(l_full_deriv, l_vertex_shader_result[0].m_proj_pos_prev, l_vertex_shader_result[1].m_proj_pos_prev, l_vertex_shader_result[2].m_proj_pos_prev);
#endif

    float2 l_uv             = float2(l_uv_x_deriv[0], l_uv_y_deriv[0]);
    float2 l_uv_ddx         = float2(l_uv_x_deriv[1], l_uv_y_deriv[1]);
    float2 l_uv_ddy         = float2(l_uv_x_deriv[2], l_uv_y_deriv[2]);

    float3 l_pos_ls         = float3(l_pos_ls_x_deriv[0], l_pos_ls_y_deriv[0], l_pos_ls_z_deriv[0]);
    float3 l_pos_ls_ddx     = float3(l_pos_ls_x_deriv[1], l_pos_ls_y_deriv[1], l_pos_ls_z_deriv[1]);
    float3 l_pos_ls_ddy     = float3(l_pos_ls_x_deriv[2], l_pos_ls_y_deriv[2], l_pos_ls_z_deriv[2]);

    ps_input_lighting_t l_ps_lighting_input = (ps_input_lighting_t)0;
    l_ps_lighting_input.m_position_ps = l_pos_ps;
    l_ps_lighting_input.m_normal = l_normal;

#if (GENERATE_TBN == 0)
    l_ps_lighting_input.m_tangent = l_tangent;
    l_ps_lighting_input.m_binormal = l_binormal;
#endif

    l_ps_lighting_input.m_texcoord0 = l_uv;
    l_ps_lighting_input.m_instance_id = l_instance_id;
    l_ps_lighting_input.m_position_ws_local = l_pos_ls;

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    l_ps_lighting_input.m_proj_pos_curr = l_proj_pos_curr;
    l_ps_lighting_input.m_proj_pos_prev = l_proj_pos_prev;
#endif

    // Get material
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[l_render_item.m_material_index];

    float4 l_direct_lighting_result = (float4)0;
    float4 l_indirect_lighting_result = (float4)0;

    lighting_pixel_shader(
        l_ps_lighting_input,
        l_camera,
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
        l_front_face,
        l_pbr_material,
        l_pos_ls_ddx,
        l_pos_ls_ddy,
        l_uv_ddx,
        l_uv_ddy,
#if defined(DEBUG_LOD)
        l_render_item.m_lod_index,
#endif
        l_direct_lighting_result,
        l_indirect_lighting_result);

    l_direct_lighting_rt[l_pixel.xy] = l_direct_lighting_result;
    l_indirect_lighting_rt[l_pixel.xy] = l_indirect_lighting_result;

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    l_velocity_rt[l_pixel.xy] = get_motion_vector_without_jitter(float2(l_camera.m_resolution_x, l_camera.m_resolution_y), l_proj_pos_curr, l_proj_pos_prev, l_camera.m_jitter, l_camera.m_jitter_prev);
#endif
}
