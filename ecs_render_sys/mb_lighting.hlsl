// Copyright:   PlayerUnknown Productions BV

#ifndef MB_LIGHTING_COMMON_H
#define MB_LIGHTING_COMMON_H

#include "mb_lighting_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"
#include "../helper_shaders/mb_wind.hlsl"

// TEMPORARY // DIRECT LIGHT WITH TRANSLUCENCY APPROXIMATION //

#if defined(MB_COMPUTE)
#   define PIXEL_SHADER_INPUT(type_and_name, semantic) type_and_name
#   define PIXEL_SHADER_INPUT_NOINTERP(type_and_name, semantic) type_and_name
#else
#   define PIXEL_SHADER_INPUT(type_and_name, semantic) type_and_name : semantic
#   define PIXEL_SHADER_INPUT_NOINTERP(type_and_name, semantic) nointerpolation type_and_name : semantic
#endif

struct ps_input_lighting_quadtree_t
{
    PIXEL_SHADER_INPUT(float4 m_position_ps         , SV_POSITION);
    PIXEL_SHADER_INPUT(float2 m_position_local      , TEXCOORD0);
    PIXEL_SHADER_INPUT(float3 m_planet_normal_ws    , TEXCOORD1);
    PIXEL_SHADER_INPUT(float3 m_surface_normal_ws   , TEXCOORD2);
    PIXEL_SHADER_INPUT(float3 m_position_ws_local   , TEXCOORD3);
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    PIXEL_SHADER_INPUT(float3 m_proj_pos_curr       , POSITION0);
    PIXEL_SHADER_INPUT(float3 m_proj_pos_prev       , POSITION1);
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    PIXEL_SHADER_INPUT(uint m_entity_id             , TEXCOORD4);
#endif
    PIXEL_SHADER_INPUT(uint m_instance_id           , TEXCOORD5);
    PIXEL_SHADER_INPUT(float m_blend_mask           , TEXCOORD6);
};

struct ps_input_lighting_t
{
    PIXEL_SHADER_INPUT(             float4 m_position_ps          , SV_POSITION);
    PIXEL_SHADER_INPUT(             float3 m_normal               , NORMAL0);
    PIXEL_SHADER_INPUT(             float2 m_texcoord0            , TEXCOORD0);
    PIXEL_SHADER_INPUT(             float3 m_position_ws_local    , POSITION0);
#if (GENERATE_TBN == 0)
    PIXEL_SHADER_INPUT(             float3 m_tangent              , TANGENT0);
    PIXEL_SHADER_INPUT(             float3 m_binormal             , BINORMAL0);
#endif
    PIXEL_SHADER_INPUT_NOINTERP(    uint m_instance_id            , ID0);
    PIXEL_SHADER_INPUT_NOINTERP(    uint m_material_buffer_srv    , ID1);
    PIXEL_SHADER_INPUT_NOINTERP(    uint m_material_index         , ID2);
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    PIXEL_SHADER_INPUT(             float3 m_proj_pos_curr        , POSITION1);
    PIXEL_SHADER_INPUT(             float3 m_proj_pos_prev        , POSITION2);
#endif
#if defined(MB_ALPHA_TEST)
    PIXEL_SHADER_INPUT_NOINTERP(    uint m_alpha_mask_texture_srv , ID3);
    PIXEL_SHADER_INPUT_NOINTERP(    float m_alpha_cutoff          , ID4);
    PIXEL_SHADER_INPUT_NOINTERP(    float m_mip_lod_bias          , ID5);
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    PIXEL_SHADER_INPUT(             uint m_entity_id              , TEXCOORD3);
#endif
};

float4 sample_texture_with_feedback(
    uint p_texture_srv,
    uint p_residency_buffer_srv,
    uint p_sampler_feedback_uav,
    uint2 p_residency_buffer_dim,
    float2 p_uv,
    float4 p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float2 p_uv_ddx,
    float2 p_uv_ddy,
#else
    float p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    float4 p_default_val = 0)
{
    float4 l_result = (float4)0;

#if defined(MB_SCALARIZE_PBR_MATERIAL_SAMPLING)
    MB_SCALARIZE_START(p_texture_srv)
#endif

    SamplerState l_sampler = (SamplerState)SamplerDescriptorHeap[SAMPLER_ANISO16_WRAP];
    if (p_residency_buffer_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_result = bindless_tex2d_sample_with_feedback(
            p_texture_srv,
            p_residency_buffer_srv,
            p_sampler_feedback_uav,
            p_residency_buffer_dim,
            l_sampler,
            p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
            p_uv_ddx,
            p_uv_ddy,
#else
            p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
            p_default_val,
            (uint2)p_position_cs.xy);
    }
    else
    {
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        l_result = bindless_tex2d_sample(p_texture_srv, l_sampler, p_uv, p_uv_ddx, p_uv_ddy, p_default_val);
#else
        l_result = bindless_tex2d_sample_bias(p_texture_srv, l_sampler, p_uv, p_mip_lod_bias, p_default_val);
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    }

#if defined(MB_SCALARIZE_PBR_MATERIAL_SAMPLING)
    MB_SCALARIZE_END
#endif

    return l_result;
}

float4 sample_texture(
    uint p_texture_srv,
    float2 p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float2 p_uv_ddx,
    float2 p_uv_ddy,
#else
    float p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    float4 p_default_val = 0)
{
    float4 l_result = (float4)0;

#if defined(MB_SCALARIZE_PBR_MATERIAL_SAMPLING)
    MB_SCALARIZE_START(p_texture_srv)
#endif

    SamplerState l_sampler = (SamplerState)SamplerDescriptorHeap[SAMPLER_ANISO16_WRAP];
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    l_result = bindless_tex2d_sample(p_texture_srv, l_sampler, p_uv, p_uv_ddx, p_uv_ddy, p_default_val);
#else
    l_result = bindless_tex2d_sample_bias(p_texture_srv, l_sampler, p_uv, p_mip_lod_bias, p_default_val);
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE

#if defined(MB_SCALARIZE_PBR_MATERIAL_SAMPLING)
    MB_SCALARIZE_END
#endif

    return l_result;
}

void get_pbr_parameters(
    sb_geometry_pbr_material_t p_pbr_material,
    float2 p_uv,
    float4 p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float2 p_uv_ddx,
    float2 p_uv_ddy,
#else
    float p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    out float4 p_base_color_texture,
    out float2 p_normal_texture,
    out float2 p_metallic_roughness,
    out float4 p_ao_texture)
{
#if defined(MB_BASE_COLOR_TEXTURE_IS_TILED) || defined(MB_TEST_ALL_PBR_TEXTURES_FOR_TILED_RESOURCES)
    p_base_color_texture = sample_texture_with_feedback(
        p_pbr_material.m_base_color_texture_srv,                           //scalarized
        p_pbr_material.m_base_color_residency_buffer_srv,
        p_pbr_material.m_base_color_sampler_feedback_uav,
        p_pbr_material.m_base_color_residency_buffer_dim,
        p_uv,
        p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f));
#else
    p_base_color_texture = sample_texture(
        p_pbr_material.m_base_color_texture_srv, //scalarized
        p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f));
#endif

#if defined(MB_NORMAL_TEXTURE_IS_TILED) || defined(MB_TEST_ALL_PBR_TEXTURES_FOR_TILED_RESOURCES)
    p_normal_texture = sample_texture_with_feedback(
        p_pbr_material.m_normal_map_texture_srv,
        p_pbr_material.m_normal_map_residency_buffer_srv,
        p_pbr_material.m_normal_map_sampler_feedback_uav,
        p_pbr_material.m_normal_map_residency_buffer_dim,
        p_uv,
        p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(0.5f, 0.5f, 1.0, 0)).rg;
#else
    p_normal_texture = sample_texture(
        p_pbr_material.m_normal_map_texture_srv,
        p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(0.5f, 0.5f, 1.0, 0)).rg;
#endif

#if defined(MB_METALLIC_ROUGHNESS_TEXTURE_IS_TILED) || defined(MB_TEST_ALL_PBR_TEXTURES_FOR_TILED_RESOURCES)
    p_metallic_roughness = sample_texture_with_feedback(
        p_pbr_material.m_metallic_roughness_texture_srv,
        p_pbr_material.m_metallic_roughness_residency_buffer_srv,
        p_pbr_material.m_metallic_roughness_sampler_feedback_uav,
        p_pbr_material.m_metallic_roughness_residency_buffer_dim,
        p_uv,
        p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f)).bg;
#else
    p_metallic_roughness = sample_texture(
        p_pbr_material.m_metallic_roughness_texture_srv,
        p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f)).bg;
#endif

#if defined(MB_OCCLUSION_TEXTURE_IS_TILED) || defined(MB_TEST_ALL_PBR_TEXTURES_FOR_TILED_RESOURCES)
    p_ao_texture = sample_texture_with_feedback(
        p_pbr_material.m_occlusion_texture_srv,
        p_pbr_material.m_occlusion_residency_buffer_srv,
        p_pbr_material.m_occlusion_sampler_feedback_uav,
        p_pbr_material.m_occlusion_residency_buffer_dim,
        p_uv,
        p_position_cs,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f));
#else
    p_ao_texture = sample_texture(
        p_pbr_material.m_occlusion_texture_srv,
        p_uv,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        float4(1.0f, 1.0f, 1.0f, 1.0f));
#endif
}

void calc_lighting_params(
    sb_geometry_pbr_material_t p_pbr_material,
    float3 p_position_ws_local,
    float3 p_normal,
    float2 p_uv,
#if (GENERATE_TBN == 0)
    float3 p_tangent,
    float3 p_binormal,
#endif // (GENERATE_TBN == 0)
    float3 p_camera_position_ws_local,
    bool p_front_face,
    float4 p_base_color_texture,
    float2 p_normal_texture,
    float4 p_ao_texture,
    float2 p_metallic_roughness,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float3 p_position_ddx,                              //ddx of argument passed to parameter p_position_ws_local
    float3 p_position_ddy,                              //ddy of argument passed to parameter p_position_ws_local
    float2 p_uv_ddx,                                    //ddx of argument passed to parameter p_uv
    float2 p_uv_ddy,                                    //ddy of argument passed to parameter p_uv
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    out float3 p_normal_ws,
    out float p_roughness,
    out float3 p_diffuse_reflectance,
    out float3 p_specular_f0,
    out float3 p_planet_normal,
    out float p_ao)
{
    // Base color
    float3 l_base_color = gamma_to_linear(p_pbr_material.m_base_color_factor.xyz * p_base_color_texture.xyz);

    // Planet normal
    p_planet_normal = normalize(p_position_ws_local + p_camera_position_ws_local);

    // Normal
    float3 l_normal_ts = 0;
    l_normal_ts.xy = p_normal_texture.xy * 2.0f - 1.0f;
    l_normal_ts.z = sqrt(1.0f - saturate(dot(l_normal_ts.xy, l_normal_ts.xy)));

    float3x3 l_tbn = (float3x3)0;
#if GENERATE_TBN == 1
    // When back-facing - flip UV to get correct tangent space
    l_tbn = build_tbn(
        normalize(p_normal),
        p_front_face ? p_uv : -p_uv
#   if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        , p_position_ddx
        , p_position_ddy
        , p_uv_ddx
        , p_uv_ddy
#   else
        , p_position_ws_local
#   endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        );
#else
    l_tbn = float3x3(normalize(p_tangent), normalize(p_binormal), normalize(p_normal));
#endif//GENERATE_TBN == 1

    // Flip the normal if backfacing, because normal is shared between both surfaces
    p_normal_ws = normalize(mul(l_normal_ts, l_tbn));
    p_normal_ws = p_front_face ? p_normal_ws : -p_normal_ws;

    // Roughness, metallic, ao
    float l_metallic = p_pbr_material.m_metallic_factor * p_metallic_roughness.x;
    p_roughness = p_pbr_material.m_roughness_factor * p_metallic_roughness.y;
    p_ao = p_ao_texture.r;

    // Diffuse color
    p_diffuse_reflectance = base_color_to_diffuse_reflectance(l_base_color, l_metallic);

    // Specular color
    p_specular_f0 = base_color_to_specular_f0(l_base_color, l_metallic);
}

void calc_lighting(
    float3 p_position_ws_local,
    float3 p_normal_ws,
    float p_roughness,
    float3 p_diffuse_reflectance,
    float3 p_specular_f0,
    float3 p_planet_normal,
    float p_ao,
    ConstantBuffer<cb_light_list_t> p_light_list,
    uint p_shadow_caster_count,
    uint p_shadow_caster_srv,
    float p_exposure_value,
    uint p_dfg_texture_srv,
    uint p_diffuse_ld_texture_srv,
    uint p_specular_ld_texture_srv,
    uint p_dfg_texture_size,
    uint p_specular_ld_mip_count,
    uint p_global_shadow_map_srv,
    float4x4 p_gsm_camera_view_local_proj,
    float3x3 p_align_ground_rotation,
    out float3 p_direct_lighting,
    out float3 p_indirect_lighting)
{
    // Camera-local rendering
    float3 l_view_dir = normalize(-p_position_ws_local);

    p_direct_lighting = (float3)0;
#if defined(MB_TRANSLUCENT_LIGHTING)
    // Direct lighting with translucency
    p_direct_lighting = light_direct_translucent(
        p_light_list, p_normal_ws, l_view_dir, p_roughness, p_diffuse_reflectance, p_specular_f0,
        p_position_ws_local,
        p_planet_normal,
        p_shadow_caster_count, p_shadow_caster_srv,
        p_global_shadow_map_srv, p_gsm_camera_view_local_proj);
#else
    // Direct lighting
    p_direct_lighting = light_direct(
        p_light_list, p_normal_ws, l_view_dir, p_roughness, p_diffuse_reflectance, p_specular_f0,
        p_position_ws_local,
        p_planet_normal,
        p_shadow_caster_count, p_shadow_caster_srv,
        p_global_shadow_map_srv, p_gsm_camera_view_local_proj);
#endif

    p_indirect_lighting = (float3)0;
#if ENABLE_IBL
    p_indirect_lighting = light_ibl(
        p_normal_ws, l_view_dir, p_roughness, p_ao, p_diffuse_reflectance, p_specular_f0,
        p_exposure_value,
        p_dfg_texture_srv, p_diffuse_ld_texture_srv, p_specular_ld_texture_srv,
        p_dfg_texture_size, p_specular_ld_mip_count,
        p_light_list, p_planet_normal, (float3x3)p_align_ground_rotation);
#endif
}

void calc_lighting_params_quadtree(
    terrain_material_t p_material,
    float3 p_position_ws_local,         //p_input.m_position_ws_local
    float3 p_surface_normal_ws,         //p_input.m_surface_normal_ws
    float2 p_position_local,            //l_position_local
    float p_metallic,                   //l_metallic
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float3 p_position_ddx,              //l_pos_ls_ddx
    float3 p_position_ddy,              //l_pos_ls_ddy
    float2 p_uv_ddx,                    //l_uv_ddx
    float2 p_uv_ddy,                    //l_uv_ddy
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    out float3 p_normal_ws,
    out float p_roughness,
    out float3 p_diffuse_reflectance,
    out float3 p_specular_f0,
    out float p_ao)
{
    float3x3 l_tbn = build_tbn(	normalize(p_surface_normal_ws),
                                p_position_local
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
                                , p_position_ddx
                                , p_position_ddy
                                , p_uv_ddx
                                , p_uv_ddy
#else
                                , p_position_ws_local
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                );
    p_normal_ws = normalize(mul(p_material.m_normal_ts, l_tbn));
    p_roughness = p_material.m_roughness;

    // Diffuse color
    p_diffuse_reflectance = base_color_to_diffuse_reflectance(p_material.m_base_color, p_metallic);
    // Specular color
    p_specular_f0 = base_color_to_specular_f0(p_material.m_base_color, p_metallic);

    p_ao = p_material.m_ao;
}

bool alpha_test(
    float p_opacity,
    float3 p_position_ws_local)
{
    float l_clip_fade = saturate(length(p_position_ws_local) / MB_CLIP_RANGE_END) * MB_CLIP_GRADIENT_MAX;
    return (p_opacity - (1.f - l_clip_fade)) < 0.f;
}

void lighting_pixel_shader_quadtree(
    ps_input_lighting_quadtree_t p_input,
    sb_quadtree_material_t p_quadtree_material,
    sb_tile_instance_t p_tile,
    uint p_shadow_caster_count,
    uint p_shadow_caster_srv,
    uint p_gsm_srv,
    float4x4 p_gsm_camera_view_local_proj,
    float p_exposure_value,
    uint p_dfg_texture_srv,
    uint p_diffuse_ld_texture_srv,
    uint p_specular_ld_texture_srv,
    uint p_dfg_texture_size,
    uint p_specular_ld_mip_count,
    uint p_light_list_cbv,
    float3x3 p_align_ground_rotation,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float3 p_position_ddx,                              //ddx of argument passed to parameter p_position_ws_local
    float3 p_position_ddy,                              //ddy of argument passed to parameter p_position_ws_local
    float2 p_uv_ddx,                                    //ddx of argument passed to parameter p_uv
    float2 p_uv_ddy,                                    //ddy of argument passed to parameter p_uv
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    out float4 p_direct_lighting_result,
    out float4 p_indirect_lighting_result)
{
    float2 l_tile_uv = tile_position_to_tile_uv(p_input.m_position_local);

    // Get tile's material
    terrain_material_t l_material = sample_terrain_material(p_tile, p_quadtree_material, l_tile_uv, p_input.m_position_ws_local);

    // Get parent material and blend
#if defined(TERRAIN_BLENDING)
    blend_with_parent(l_material, p_tile, p_quadtree_material, p_input.m_position_local, p_input.m_blend_mask, p_input.m_position_ws_local);
#endif

    // Lerp tangent space normal to "up" direction
    // In the last mip levels avg normal can be different from (0, 0, 1)
    // It can result in lighting gaps on the edge of the quadtree
    l_material.m_normal_ts = lerp(float3(0, 0, 1.0f), l_material.m_normal_ts, saturate((p_tile.m_basic_data.m_tile_level - 12.0f) / (2.0f)));
    l_material.m_normal_ts = normalize(l_material.m_normal_ts);

    float l_metallic = MB_QUADTREE_METALLIC_DEFAULT;

    float3 l_normal_ws = (float3)0;
    float  l_roughness = 0;
    float3 l_diffuse_reflectance = (float3)0;
    float3 l_specular_f0 = (float3)0;
    float  l_ao = 0;
    calc_lighting_params_quadtree(  l_material,
                                    p_input.m_position_ws_local,
                                    p_input.m_surface_normal_ws,
                                    p_input.m_position_local,
                                    l_metallic,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
                                    p_position_ddx,
                                    p_position_ddy,
                                    p_uv_ddx,
                                    p_uv_ddy,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                    l_normal_ws,
                                    l_roughness,
                                    l_diffuse_reflectance,
                                    l_specular_f0,
                                    l_ao);

    float3 l_direct_lighting = (float3)0;
    float3 l_indirect_lighting = (float3)0;
    calc_lighting(
        p_input.m_position_ws_local,
        l_normal_ws,
        l_roughness,
        l_diffuse_reflectance,
        l_specular_f0,
        p_input.m_planet_normal_ws,
        l_ao,
        ResourceDescriptorHeap[p_light_list_cbv],
        p_shadow_caster_count,
        p_shadow_caster_srv,
        p_exposure_value,
        p_dfg_texture_srv,
        p_diffuse_ld_texture_srv,
        p_specular_ld_texture_srv,
        p_dfg_texture_size,
        p_specular_ld_mip_count,
        p_gsm_srv,
        p_gsm_camera_view_local_proj,
        p_align_ground_rotation,
        l_direct_lighting,
        l_indirect_lighting);

    p_direct_lighting_result = float4(pack_lighting(l_direct_lighting), 1);
    p_indirect_lighting_result = float4(pack_lighting(l_indirect_lighting), 0);

    // When raytracing is used for diffuse GI - no need to compute indirect lighting
    // However, we need diffuse reflectance to multiply indirect lighting computed with DXR
    // Primary rays are not traced, so we need diffuse reflectance from V-Buffer pass
#if defined(MB_RAYTRACING_DIFFUSE_GI)
    p_indirect_lighting_result = float4(l_diffuse_reflectance, 0);
#endif

    //! \todo The debug code should be made only available in builds with debug functionality
    get_quadtree_debug_color(p_quadtree_material, p_tile, l_material, p_input.m_position_local, p_input.m_blend_mask, p_direct_lighting_result, p_indirect_lighting_result);
}

#define NUM_LOD_DEBUG_COLORS 5
static const float3 lod_debug_colors[NUM_LOD_DEBUG_COLORS] =
{
    float3(1.0f, 0.0f, 0.0f),   // Red
    float3(0.0f, 1.0f, 0.0f),   // Green
    float3(0.0f, 0.0f, 1.0f),   // Blue
    float3(1.0f, 1.0f, 0.0f),   // Yellow
    float3(0.56f, 0.0f, 1.0f),  // Violet
};

bool lighting_pixel_shader(
    ps_input_lighting_t p_input,
    ConstantBuffer<cb_camera_t> p_camera,
    uint p_shadow_caster_count,
    uint p_shadow_caster_srv,
    uint p_gsm_srv,
    float4x4 p_gsm_camera_view_local_proj,
    float p_exposure_value,
    uint p_dfg_texture_srv,
    uint p_diffuse_ld_texture_srv,
    uint p_specular_ld_texture_srv,
    uint p_dfg_texture_size,
    uint p_specular_ld_mip_count,
    uint p_light_list_cbv,
    bool p_front_face,
    sb_geometry_pbr_material_t p_pbr_material,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
    float3 p_position_ddx,                              //ddx of argument passed to parameter p_position_ws_local
    float3 p_position_ddy,                              //ddy of argument passed to parameter p_position_ws_local
    float2 p_uv_ddx,                                    //ddx of argument passed to parameter p_uv
    float2 p_uv_ddy,                                    //ddy of argument passed to parameter p_uv
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
#if defined(DEBUG_LOD)
    uint p_lod_index,
#endif
    out float4 p_direct_lighting_result,
    out float4 p_indirect_lighting_result)
{
    float4 l_base_color_texture = (float4)0;
    float2 l_normal_texture = (float2)0;
    float2 l_metallic_roughness = (float2)0;
    float4 l_ao_texture = (float4)0;
    get_pbr_parameters(
        p_pbr_material,
        p_input.m_texcoord0,
        p_input.m_position_ps,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_uv_ddx,
        p_uv_ddy,
#else
        p_camera.m_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        l_base_color_texture,
        l_normal_texture,
        l_metallic_roughness,
        l_ao_texture);

#if defined(MB_ALPHA_TEST)
    if(alpha_test(l_base_color_texture.a, p_input.m_position_ws_local))
    {
        return false;
    }
#endif

    float3 l_normal_ws = (float3)0;
    float  l_roughness = 0;
    float3 l_diffuse_reflectance = (float3)0;
    float3 l_specular_f0 = (float3)0;
    float3 l_planet_normal = (float3)0;
    float l_ao = 0;
    calc_lighting_params(
        p_pbr_material,
        p_input.m_position_ws_local,
        p_input.m_normal,
        p_input.m_texcoord0,
#if (GENERATE_TBN == 0)
        p_input.m_tangent,
        p_input.m_binormal,
#endif //(GENERATE_TBN == 0)
        p_camera.m_camera_pos,
        p_front_face,
        l_base_color_texture,
        l_normal_texture,
        l_ao_texture,
        l_metallic_roughness,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
        p_position_ddx,                              //ddx of argument passed to parameter p_position_ws_local
        p_position_ddy,                              //ddy of argument passed to parameter p_position_ws_local
        p_uv_ddx,                                    //ddx of argument passed to parameter p_uv
        p_uv_ddy,                                    //ddy of argument passed to parameter p_uv
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        l_normal_ws,
        l_roughness,
        l_diffuse_reflectance,
        l_specular_f0,
        l_planet_normal,
        l_ao);

    float3 l_direct_lighting = (float3)0;
    float3 l_indirect_lighting = (float3)0;
    calc_lighting(
        p_input.m_position_ws_local,
        l_normal_ws,
        l_roughness,
        l_diffuse_reflectance,
        l_specular_f0,
        l_planet_normal,
        l_ao,
        ResourceDescriptorHeap[p_light_list_cbv],
        p_shadow_caster_count,
        p_shadow_caster_srv,
        p_exposure_value,
        p_dfg_texture_srv,
        p_diffuse_ld_texture_srv,
        p_specular_ld_texture_srv,
        p_dfg_texture_size,
        p_specular_ld_mip_count,
        p_gsm_srv,
        p_gsm_camera_view_local_proj,
        (float3x3)p_camera.m_align_ground_rotation,
        l_direct_lighting,
        l_indirect_lighting);

    // Save lighting results
    p_direct_lighting_result = float4(l_direct_lighting, l_base_color_texture.a);
    p_indirect_lighting_result = float4(l_indirect_lighting, 0.0f);

#if DEBUG_DIFFUSE
    p_direct_lighting_result = float4(linear_to_gamma(l_diffuse_reflectance), 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_NORMAL
    p_direct_lighting_result = float4(l_normal_ws, 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_SPECULAR
    p_direct_lighting_result = float4(linear_to_gamma(l_specular_f0), 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_ROUGHNESS
    p_direct_lighting_result = float4(l_roughness.xxx, 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_METALLIC
    p_direct_lighting_result = float4(l_metallic.xxx, 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_OCCLUSION
    p_direct_lighting_result = float4(l_ao.xxx, 1.0f);
    p_indirect_lighting_result = 0;
#elif DEBUG_LOD
    float3 l_lod_color = lod_debug_colors[clamp(p_lod_index, 0, NUM_LOD_DEBUG_COLORS - 1)];
    p_direct_lighting_result = float4(unpack_lighting(l_lod_color), 1.0f);
    p_indirect_lighting_result = 0;
#endif

    p_direct_lighting_result = float4(pack_lighting(p_direct_lighting_result.rgb), l_base_color_texture.a);
    p_indirect_lighting_result = float4(pack_lighting(p_indirect_lighting_result.rgb), 0.0f);

    // When raytracing is used for diffuse GI - no need to compute indirect lighting
    // However, we need diffuse reflectance to multiply indirect lighting computed with DXR
    // Primary rays are not traced, so we need diffuse reflectance from V-Buffer pass
#if defined(MB_RAYTRACING_DIFFUSE_GI)
    p_indirect_lighting_result = float4(l_diffuse_reflectance, 0);
#endif

    return true;
}

ps_input_lighting_quadtree_t lighting_vertex_shader_quadtree(
    uint p_vertex_id,
    sb_render_item_t p_render_item,
    sb_quadtree_material_t p_quadtree_material,
    sb_render_instance_t p_render_instance,
    StructuredBuffer<sb_tile_instance_t> p_tile_instances,
    ConstantBuffer<cb_camera_t> p_camera,
    uint p_instance_id)
{
    sb_tile_instance_t l_tile = p_tile_instances[p_render_instance.m_user_data];

    // Get vertex resolution
    uint l_vertex_resolution = p_quadtree_material.m_tile_size_in_vertices;

    // Get tile position from vertex id
    float2 l_tile_position = get_tile_position(p_render_item, p_vertex_id, l_vertex_resolution);

    float l_blend_mask = 0;
#if defined(TERRAIN_BLENDING)
    bool l_position_moved = false;

    // Move vertices to match neighboring tiles
    terrain_blend_mask_vertex(l_tile,
                              l_vertex_resolution,
                              l_tile_position,
                              l_position_moved,
                              l_blend_mask);

    // Get mesh data
    terrain_sample_t l_terrain_sample = sample_terrain(p_render_item, l_tile, l_tile_position, l_vertex_resolution, l_position_moved);

    // Merge vertices to match parent-tile vertex frequency
    if (l_tile.m_parent_index != TILE_NO_PARENT)
    {
        blend_with_parent(l_terrain_sample,
                          l_tile_position,
                          l_tile,
                          p_render_item,
                          p_tile_instances[l_tile.m_parent_index],
                          l_vertex_resolution,
                          l_blend_mask,
                          l_position_moved);
    }
#else
    // Get mesh data
    terrain_sample_t l_terrain_sample = sample_terrain(p_render_item, l_tile, l_tile_position, l_vertex_resolution);
#endif

#if defined(ENABLE_SKIRT)
    apply_skirt(l_tile_position,
                p_quadtree_material.m_skirt_distance_threshold_squared,
                p_quadtree_material.m_skirt_scale,
                p_render_item,
                l_tile,
                l_vertex_resolution,
                l_terrain_sample.m_position_ws_local);
#endif

    // Compute positions
    float4 l_pos_ws_local = float4(l_terrain_sample.m_position_ws_local, 1.0);
    float4 l_pos_vs = mul(l_pos_ws_local, p_camera.m_view_local);
    float4 l_pos_ps = mul(l_pos_vs, p_camera.m_proj);

    // Vertex shader output
    ps_input_lighting_quadtree_t l_result;
    l_result.m_position_ps          = l_pos_ps;
    l_result.m_position_local       = l_tile_position;
    l_result.m_planet_normal_ws     = l_terrain_sample.m_planet_normal_ws;
    l_result.m_surface_normal_ws    = l_terrain_sample.m_surface_normal_ws;
    l_result.m_position_ws_local    = l_terrain_sample.m_position_ws_local;
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float4 l_pos_ps_prev            = mul(l_pos_ws_local, p_camera.m_view_proj_local_prev);

    l_result.m_proj_pos_curr        = l_pos_ps.xyw;
    l_result.m_proj_pos_prev        = l_pos_ps_prev.xyw;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_result.m_entity_id            = l_tile.m_basic_data.m_entity_id;
#endif
    l_result.m_instance_id          = p_instance_id;
    l_result.m_blend_mask           = l_blend_mask;

    return l_result;
}

ps_input_lighting_t lighting_vertex_shader(
    uint p_index,
    sb_render_item_t p_render_item,
    sb_render_instance_t p_render_instance,
    bool p_wind,
    bool p_wind_small,
    uint p_time,
    uint p_time_prev,
    ConstantBuffer<cb_camera_t> p_camera,
    uint p_instance_id)
{
    mesh_vertex_t l_mesh_vertex = (mesh_vertex_t)0;
    get_vertex_mesh_position(p_index, p_render_item, l_mesh_vertex);
    get_vertex_mesh_other(p_index, p_render_item, l_mesh_vertex);

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float3 l_vertex_position_prev = l_mesh_vertex.m_position;
    if(p_wind)
    {
        l_vertex_position_prev = apply_wind_to_position(l_vertex_position_prev, p_render_instance.m_transform, p_time_prev, p_camera, p_wind_small);
    }
#endif

    if(p_wind)
    {
        l_mesh_vertex.m_position = apply_wind_to_position(l_mesh_vertex.m_position, p_render_instance.m_transform, p_time, p_camera, p_wind_small);
    }

    // POSITION: Transform to camera local space
    float3 l_pos_ls = mul(float4(l_mesh_vertex.m_position.xyz, 1.0f), p_render_instance.m_transform);

    float4x4 l_vp = mul(p_camera.m_view_local, p_camera.m_proj);
    float4 l_pos_ps = mul(float4(l_pos_ls, 1.0f), l_vp);

    // NORMAL: Transform to camera local space
    float3 l_normal_ws = normalize(mul(l_mesh_vertex.m_normal, (float3x3)p_render_instance.m_transform));

    // Vertex shader output
    ps_input_lighting_t l_result = (ps_input_lighting_t)0;
    l_result.m_position_ps          = l_pos_ps;
    l_result.m_normal               = l_normal_ws;     // Does not support non-uniform scale
#if (GENERATE_TBN == 0)
    // TANGENT: Transform to camera local space
    float3 l_tangent_ws             = normalize(mul(l_mesh_vertex.m_tangent.xyz, (float3x3)p_render_instance.m_transform));
    l_result.m_tangent              = l_tangent_ws;    // Does not support non-uniform scale
    l_result.m_binormal             = cross(l_tangent_ws, l_normal_ws) * l_mesh_vertex.m_tangent.w;
#endif
    l_result.m_texcoord0            = l_mesh_vertex.m_uv0;
    l_result.m_position_ws_local    = l_pos_ls;
    l_result.m_instance_id          = p_instance_id;
    l_result.m_material_buffer_srv  = p_render_item.m_material_buffer_srv;
    l_result.m_material_index       = p_render_item.m_material_index;
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float3 l_pos_ls_prev = mul(float4(l_vertex_position_prev, 1.0f), p_render_instance.m_transform_prev);
    float4 l_pos_ps_prev = mul(float4(l_pos_ls_prev, 1.0f), p_camera.m_view_proj_local_prev);

    l_result.m_proj_pos_curr = l_pos_ps.xyw;
    l_result.m_proj_pos_prev = l_pos_ps_prev.xyw;
#endif
#if defined(MB_ALPHA_TEST)
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[p_render_item.m_material_buffer_srv];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[p_render_item.m_material_index];

    l_result.m_alpha_mask_texture_srv = l_pbr_material.m_alpha_mask_texture_srv;
    l_result.m_alpha_cutoff           = l_pbr_material.m_alpha_cutoff;
    l_result.m_mip_lod_bias           = p_camera.m_mip_lod_bias;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_result.m_entity_id            = p_render_instance.m_entity_id;
#endif

    return l_result;
}

#endif //MB_LIGHTING_COMMON_H
