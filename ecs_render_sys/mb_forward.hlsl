// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_gltf_t>                  g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

#define GENERATE_TBN (1)

// Helper functions
#include "mb_lighting.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------
ps_input_lighting_t vs_gpu_instancing(  uint p_vertex_id    : SV_VertexID,
                                        uint p_instance_id  : SV_InstanceID)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Unpack mesh
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    // Unpack input data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

#if defined(MB_WIND)
    const bool l_wind = true;
#else
    const bool l_wind = false;
#endif

#if defined(MB_WIND_SMALL)
    const bool l_wind_small = true;
#else
    const bool l_wind_small = false;
#endif

    ps_input_lighting_t l_result = lighting_vertex_shader(
        p_vertex_id,
        l_render_item,
        l_render_instance,
        l_wind,
        l_wind_small,
        g_push_constants.m_time,
        l_camera,
        p_instance_id);

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ps_shadow_pass(ps_input_lighting_t p_input,
                    bool l_front_face : SV_IsFrontFace)
{
#if defined(MB_ALPHA_TEST)
    // Fetch alpha mask (stored in the red channel) and apply alpha testing
    float l_opacity = bindless_tex2d_sample(p_input.m_alpha_mask_texture_srv,
                                            (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], 
                                            p_input.m_texcoord0,
                                            0.0).r;

    clip(l_opacity - p_input.m_alpha_cutoff);
#endif
}

void ps_visibility_pass()
{
}


//-----------------------------------------------------------------------------
struct ps_impostor_data_t
{
    float4 m_albedo_alpha : SV_TARGET0;
    float4 m_normal_depth : SV_TARGET1;
};

//-----------------------------------------------------------------------------
ps_impostor_data_t ps_impostor_data_pass(ps_input_lighting_t p_input,
                                         bool p_front_face : SV_IsFrontFace)
{
    // Get camera
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get material
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(p_input.m_material_buffer_srv)];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[p_input.m_material_index];

    float4 l_base_color_texture = (float4)0;
    float2 l_normal_texture = (float2)0;
    float2 l_metallic_roughness = (float2)0;
    float4 l_ao_texture = (float4)0;
    get_pbr_parameters(
        l_pbr_material, 
        p_input.m_texcoord0, 
        p_input.m_position_ps, 
        l_base_color_texture, 
        l_normal_texture, 
        l_metallic_roughness, 
        l_ao_texture);

    float3 l_normal_ws = (float3)0;
    float  l_roughness = 0;
    float3 l_diffuse_reflectance = (float3)0;
    float3 l_specular_f0 = (float3)0;
    float3 l_planet_normal = (float3)0;
    float l_ao = 0;
    calc_lighting_params(
        l_pbr_material,
        p_input.m_position_ws_local,
        p_input.m_normal,
        p_input.m_texcoord0,
#if (GENERATE_TBN == 0)
        p_input.m_tangent,
        p_input.m_binormal,
#endif //(GENERATE_TBN == 0)
        l_camera.m_camera_pos,
        p_front_face,
        l_base_color_texture,
        l_normal_texture,
        l_ao_texture,
        l_metallic_roughness,
        l_normal_ws,
        l_roughness,
        l_diffuse_reflectance,
        l_specular_f0,
        l_planet_normal,
        l_ao);

#if defined(MB_ALPHA_TEST)
    if (alpha_test(l_base_color_texture.a, p_input.m_position_ws_local))
    {
        discard;
    }
#endif

    ps_impostor_data_t l_output;

    l_output.m_albedo_alpha = l_base_color_texture;
    l_output.m_normal_depth = float4(l_normal_ws*0.5f + 0.5f, p_input.m_position_ps.z);

    return l_output;
}


//-----------------------------------------------------------------------------
ps_output_t ps_main(ps_input_lighting_t p_input,
                    bool p_front_face : SV_IsFrontFace)
{
    // Get camera
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get material
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[p_input.m_material_buffer_srv];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[p_input.m_material_index];

#if defined(DEBUG_LOD) // These are only required for the LOD level debug view
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_input.m_instance_id];

    StructuredBuffer<sb_render_item_t> l_render_items = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items[l_render_instance.m_render_item_idx];
#endif

    float4 l_direct_lighting_result = (float4)0;
    float4 l_indirect_lighting_result = (float4)0;

    if(!lighting_pixel_shader(
        p_input,
        l_camera.m_camera_pos,
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
        p_front_face,
        l_pbr_material,
#if defined(DEBUG_LOD)
        l_render_item.m_lod_index,
#endif
        l_direct_lighting_result,
        l_indirect_lighting_result))
    {
        discard;
    }

    ps_output_t l_ps_output = (ps_output_t)0;
    l_ps_output.m_direct_lighting = l_direct_lighting_result;
    l_ps_output.m_indirect_lighting = l_indirect_lighting_result;

    // Pack lighting
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    l_ps_output.m_velocity = p_input.m_velocity_data.xy - p_input.m_velocity_data.zw;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_ps_output.m_entity_id = pack_entity_id(p_input.m_entity_id);
#endif

    return l_ps_output;
}

//-----------------------------------------------------------------------------
void ps_occlusion_pre_pass(ps_input_lighting_t p_input)
{
#if defined(MB_ALPHA_TEST)
    // Fetch alpha mask (stored in the red channel) and apply alpha testing
    float l_opacity = bindless_tex2d_sample(p_input.m_alpha_mask_texture_srv,
                                            (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], 
                                            p_input.m_texcoord0,
                                            0.0).r;

    clip(l_opacity - p_input.m_alpha_cutoff);
#endif
}