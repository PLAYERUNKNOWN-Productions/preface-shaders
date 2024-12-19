// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

#define GENERATE_TBN 1
#include "mb_lighting.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_gltf_t>                  g_push_constants    : register(REGISTER_PUSH_CONSTANTS);

// UAV

// Helper functions
#include "../helper_shaders/mb_wind.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct ps_input_t
{
                        float4  m_position_ps            : SV_POSITION;
#if defined(MB_ALPHA_TEST)
                        float2  m_texcoord0              : TEXCOORD0;
                        float3  m_position_ws_local      : POSITION;
#endif
    nointerpolation     uint    m_instance_id            : ID0;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    nointerpolation     uint    m_entity_id              : ID1;
#endif
#if defined(MB_ALPHA_TEST)
    nointerpolation     uint    m_alpha_mask_texture_srv : ID2;
    nointerpolation     float   m_alpha_cutoff           : ID3;
#endif
};

//-----------------------------------------------------------------------------
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
    uint l_result = e_lighting_classification_mask_default;
#if defined(MB_TRANSLUCENT_LIGHTING)
    l_result = e_lighting_classification_mask_translucent_lighting;
#endif

    return l_result;
}

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ps_input_t vs_gpu_instancing(   uint p_vertex_id    : SV_VertexID,
                                uint p_instance_id  : SV_InstanceID)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Unpack mesh
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    mesh_vertex_t l_mesh_vertex = (mesh_vertex_t)0;
    get_vertex_mesh_position(p_vertex_id, l_render_item, l_mesh_vertex);
#if defined(MB_ALPHA_TEST)
    get_vertex_mesh_other(p_vertex_id, l_render_item, l_mesh_vertex);
#endif

    // Unpack input data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

#if defined(MB_WIND_SMALL)
    const bool l_wind_small = true;
#else
    const bool l_wind_small = false;
#endif //MB_WIND_SMALL

#if defined(MB_WIND) // Flora Movement //
    l_mesh_vertex.m_position = apply_wind_to_position(l_mesh_vertex.m_position, l_render_instance.m_transform, g_push_constants.m_time, l_camera, l_wind_small);
#endif

    float3 l_pos_ws  = mul(float4(l_mesh_vertex.m_position.xyz, 1.0f), l_render_instance.m_transform);  //world space
    
    float4x4 l_vp = mul(l_camera.m_view_local, l_camera.m_proj);
    float4 l_pos_ps  = mul(float4(l_pos_ws, 1.0f), l_vp);                                               //projection space

    // Vertex shader output
    ps_input_t l_result = (ps_input_t)0;
    l_result.m_position_ps          = l_pos_ps;
#if defined(MB_ALPHA_TEST)
    l_result.m_texcoord0            = l_mesh_vertex.m_uv0;
    l_result.m_position_ws_local    = l_pos_ws;
#endif
    l_result.m_instance_id          = p_instance_id + g_push_constants.m_render_instance_buffer_offset;

#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_result.m_entity_id            = l_render_instance.m_entity_id;
#endif

#if defined(MB_ALPHA_TEST)
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[l_render_item.m_material_buffer_srv];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[l_render_item.m_material_index];
    
    l_result.m_alpha_mask_texture_srv = l_pbr_material.m_alpha_mask_texture_srv;
    l_result.m_alpha_cutoff           = l_pbr_material.m_alpha_cutoff;
#endif

    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ps_shadow_pass(ps_input_t p_input)
{
#if defined(MB_ALPHA_TEST)
    // Fetch alpha mask (stored in the red channel) and apply alpha testing
    float l_opacity = bindless_tex2d_sample(
                p_input.m_alpha_mask_texture_srv,
                (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], 
                p_input.m_texcoord0,
                0.0).r;

    clip(l_opacity - p_input.m_alpha_cutoff);
#endif
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
#if defined(MB_ALPHA_TEST)
    float4 l_base_color_texture = (float4)0;

    // Fetch alpha mask (stored in the red channel) and apply alpha testing
    l_base_color_texture.a = bindless_tex2d_sample(
                p_input.m_alpha_mask_texture_srv,
                (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], 
                p_input.m_texcoord0,
                1.0).r;

    if(alpha_test(l_base_color_texture.a, p_input.m_position_ws_local))
    {
        discard;
    }
#endif // MB_ALPHA_TEST

    ps_output_t l_ps_output = (ps_output_t)0;

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

    uint l_packed_instance_id_pixel_options = pack_instance_id_pixel_options(p_input.m_instance_id, p_front_face, l_wind, l_wind_small);
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
#if defined(MB_ALPHA_TEST)
    float4 l_base_color_texture = (float4)0;

    // Fetch alpha mask (stored in the red channel) and apply alpha testing
    l_base_color_texture.a = bindless_tex2d_sample(
                p_input.m_alpha_mask_texture_srv,
                (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], 
                p_input.m_texcoord0,
                1.0).r;

    if(alpha_test(l_base_color_texture.a, p_input.m_position_ws_local))
    {
        discard;
    }
#endif // MB_ALPHA_TEST
}