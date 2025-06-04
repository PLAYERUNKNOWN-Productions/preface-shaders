// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

#define time_scale 0.0000001f
#define fade_out_distance 2000.0f

#define MB_FADE_OUT_WAVES

// Push constants
ConstantBuffer<cb_push_gltf_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

// Helper functions
#include "mb_lighting.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../shared_shaders/mb_ocean_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"

struct ps_output
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

struct ps_input_lighting_ocean
{
    float4 m_position_ps        : SV_POSITION;
    float2 m_position_local     : TEXCOORD0;
    float3 m_planet_normal_ws   : TEXCOORD1;
    float3 m_surface_normal_ws  : TEXCOORD2;
    float3 m_position_ws_local  : TEXCOORD3;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    uint m_entity_id            : TEXCOORD4;
#endif
    uint m_instance_id          : TEXCOORD5;
    float m_blend_mask          : TEXCOORD6;
};

float3 get_planet_normal(sb_tile_instance_base tile, float2 tile_position)
{
    float3 normal_01 = lerp(tile.m_normal_0.xyz, tile.m_normal_1.xyz, tile_position.x);
    float3 normal_23 = lerp(tile.m_normal_3.xyz, tile.m_normal_2.xyz, tile_position.x);
    float3 normal = lerp(normal_01, normal_23, tile_position.y);
    return normalize(normal);
}

float2 get_fractional_uv(float3 tile_pos_frac, float2 tile_position)
{
    float2 frac_uv = frac(tile_pos_frac.xy + tile_position.xy * tile_pos_frac.zz);
    return frac_uv;
}

float3 get_shared_border(sb_ocean_tile_instance_t tile, float2 tile_position, uint border_index)
{
    float2 remapped_tile_position = remapTilePosition(tile_position, tile.m_shared_border_weight_masks[border_index]);
    return get_vertex_position(tile.m_neighbour_patches[border_index], remapped_tile_position);
}

float3 get_vertex_position(sb_ocean_tile_instance_t tile, float2 tile_position)
{
    // Get basic vertex position on sphere from quad patch
    float3 position = (float3)0;

    if (tile_position.x == 0 && tile_position.y == 1 && (tile.m_shared_vertices_masks & (1 << 0)))      // Top left corner
    {
        position = tile.m_shared_corners[0];
    }
    else if (tile_position.x == 1 && tile_position.y == 1 && (tile.m_shared_vertices_masks & (1 << 1))) // Top right corner
    {
        position = tile.m_shared_corners[1];
    }
    else if (tile_position.x == 0 && tile_position.y == 0 && (tile.m_shared_vertices_masks & (1 << 2))) // Bottom left corner
    {
        position = tile.m_shared_corners[2];
    }
    else if (tile_position.x == 1 && tile_position.y == 0 && (tile.m_shared_vertices_masks & (1 << 3))) // Bottom right corner
    {
        position = tile.m_shared_corners[3];
    }
    else if (tile_position.x == 0 && (tile.m_shared_vertices_masks & (1 << 4))) // Left border
    {
        position = get_shared_border(tile, tile_position, 0);
    }
    else if (tile_position.x == 1 && (tile.m_shared_vertices_masks & (1 << 5))) // Right border
    {
        position = get_shared_border(tile, tile_position, 1);
    }
    else if (tile_position.y == 0 && (tile.m_shared_vertices_masks & (1 << 6))) // Bottom border
    {
        position = get_shared_border(tile, tile_position, 2);
    }
    else if (tile_position.y == 1 && (tile.m_shared_vertices_masks & (1 << 7))) // Top border
    {
        position = get_shared_border(tile, tile_position, 3);
    }
    else
    {
        position = get_vertex_position(tile.m_tile_data.m_basic_data.m_patch_data, tile_position);
    }

    return position;
}

float get_sine_wave_height(float2 tile_pos_frac, float2 direction, float wave_length, float speed, float amplitude, float time, float phase_offset)
{
    float k = 2.0f * M_PI / wave_length;
    float omega = k * speed;

    float wave_phase = k * dot(tile_pos_frac, direction) - omega * time + phase_offset;
    return amplitude * sin(wave_phase);
}

float3 get_wave_displacements(float2 tile_pos_frac, float3 planet_normal, float time)
{
    float3 displacement = float3(0, 0, 0);

    displacement += planet_normal * fbm_repeat(tile_pos_frac + time * 10.0f, 1000, 10);
    displacement +=
        planet_normal * get_sine_wave_height(tile_pos_frac, normalize(float2(-1.0f, -1.0f)), 0.01f, 20.0f, 10.0f, time, 0.5f);

    return displacement;
}

float get_fade_out_factor(float3 position_camera_local)
{
    return saturate(length(position_camera_local) / fade_out_distance);
}

float3 get_normal_with_epsilon(sb_ocean_tile_instance_t tile, float2 tile_position, float epsilon, float time)
{
    float2 tile_position_0 = tile_position;
    float3 position_0 = get_vertex_position(tile, tile_position_0);

    float2 tile_position_1 = tile_position + float2( epsilon,     0.0f);
    float2 tile_position_2 = tile_position + float2(    0.0f,  epsilon);
    float2 tile_position_3 = tile_position + float2(-epsilon,     0.0f);
    float2 tile_position_4 = tile_position + float2(    0.0f, -epsilon);

    float3 position_1 = get_vertex_position(tile, tile_position_1);
    float3 position_2 = get_vertex_position(tile, tile_position_2);
    float3 position_3 = get_vertex_position(tile, tile_position_3);
    float3 position_4 = get_vertex_position(tile, tile_position_4);

    float3 normal_0 = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position_0);
    float3 normal_1 = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position_1);
    float3 normal_2 = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position_2);
    float3 normal_3 = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position_3);
    float3 normal_4 = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position_4);

    float2 frac_uv_0 = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position_0);
    float2 frac_uv_1 = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position_1);
    float2 frac_uv_2 = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position_2);
    float2 frac_uv_3 = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position_3);
    float2 frac_uv_4 = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position_4);

    position_0 += get_wave_displacements(frac_uv_0, normal_0, time);
    position_1 += get_wave_displacements(frac_uv_1, normal_1, time);
    position_2 += get_wave_displacements(frac_uv_2, normal_2, time);
    position_3 += get_wave_displacements(frac_uv_3, normal_3, time);
    position_4 += get_wave_displacements(frac_uv_4, normal_4, time);

    // Use cross-products to avarage normals
    float3 normal = 0;
    normal += normalize(cross(position_2 - position_0, position_1 - position_0));
    normal += normalize(cross(position_1 - position_0, position_4 - position_0));
    normal += normalize(cross(position_4 - position_0, position_3 - position_0));
    normal += normalize(cross(position_3 - position_0, position_2 - position_0));
    normal = normalize(normal);

#if defined(MB_FADE_OUT_WAVES)
    // Fade out normal with planet normal
    normal = normalize(lerp(normal, normal_0, get_fade_out_factor(position_0)));
#endif

    return normal;
}

ps_input_lighting_ocean lighting_vertex_shader_ocean(uint vertex_id,
                                                     sb_render_item_t render_item,
                                                     sb_ocean_material material,
                                                     sb_render_instance_t render_instance,
                                                     StructuredBuffer<sb_ocean_tile_instance_t> tile_instances,
                                                     ConstantBuffer<cb_camera_t> camera,
                                                     uint instance_id,
                                                     float time)
{
    sb_ocean_tile_instance_t tile = tile_instances[render_instance.m_user_data];

    float2 tile_position = get_tile_position(render_item, vertex_id, material.m_vertex_resolution);

    // T-junctions
#if defined(TERRAIN_BLENDING)
    // Move vertices to match neighboring tiles
    bool position_moved = false;
    float blend_mask = 0;
    terrain_blend_mask_vertex(tile.m_tile_data, material.m_vertex_resolution, tile_position, position_moved, blend_mask);
#endif

    float3 planet_normal = get_planet_normal(tile.m_tile_data.m_basic_data, tile_position);

    float3 position_ws_local = get_vertex_position(tile, tile_position);

    float2 frac_uv = get_fractional_uv(tile.m_tile_pos_frac_10000, tile_position);
    float3 wave_displacement = get_wave_displacements(frac_uv, planet_normal, time);

#if defined(MB_FADE_OUT_WAVES)
    // Fade out displacement by distance
    wave_displacement = lerp(wave_displacement, float3(0, 0, 0), get_fade_out_factor(position_ws_local));
#endif

    position_ws_local += wave_displacement;

    float4 pos_ws_local = float4(position_ws_local, 1.0);
    float4 pos_vs = mul(pos_ws_local, camera.m_view_local);
    float4 pos_ps = mul(pos_vs, camera.m_proj);

    // Vertex shader output
    ps_input_lighting_ocean result;
    result.m_position_ps        = pos_ps;
    result.m_position_local     = tile_position;
    result.m_planet_normal_ws   = planet_normal;
    result.m_surface_normal_ws  = (float3)0;
    result.m_position_ws_local  = position_ws_local;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    result.m_entity_id          = tile.m_tile_data.m_basic_data.m_entity_id;
#endif
    result.m_instance_id        = instance_id;
    result.m_blend_mask         = 0;

    return result;
}

void lighting_pixel_shader_ocean(ps_input_lighting_ocean input,
                                 sb_ocean_material material,
                                 sb_ocean_tile_instance_t tile,
                                 uint shadow_caster_count,
                                 uint shadow_caster_srv,
                                 uint gsm_srv,
                                 float4x4 gsm_camera_view_local_proj,
                                 float exposure_value,
                                 uint dfg_texture_srv,
                                 uint diffuse_ld_texture_srv,
                                 uint specular_ld_texture_srv,
                                 uint dfg_texture_size,
                                 uint specular_ld_mip_count,
                                 uint light_list_cbv,
                                 float3x3 align_ground_rotation,
#if defined(MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE)
                                 float3 position_ddx,                              //ddx of argument passed to parameter position_ws_local
                                 float3 position_ddy,                              //ddy of argument passed to parameter position_ws_local
                                 float2 uv_ddx,                                    //ddx of argument passed to parameter uv
                                 float2 uv_ddy,                                    //ddy of argument passed to parameter uv
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                 float time,
                                 out float4 direct_lighting_result,
                                 out float4 indirect_lighting_result)
{
    float3 normal_ws = (float3)0;
    float  roughness = 0;
    float3 diffuse_reflectance = (float3)0;
    float3 specular_f0 = (float3)0;
    float  ao = 0;

    // Hack lighting for ocean
    {
        // Scale by tile size
        const uint reference_level = 23;
        uint level_diff = max(0, reference_level - tile.m_tile_data.m_basic_data.m_tile_level);
        uint size = (1u << level_diff);
        float scale = 1.0f / size;
        float epsilon = scale / (material.m_vertex_resolution - 1);

        float3 surface_normal = get_normal_with_epsilon(tile, input.m_position_local, epsilon, time);

        normal_ws = surface_normal;
        roughness = 0;
        diffuse_reflectance = float3(0.03f, 0.07f, 0.12f);
        specular_f0 = float3(0.02f, 0.02f, 0.02f);
        ao = 1.0f;
    }

    float3 direct_lighting = (float3)0;
    float3 indirect_lighting = (float3)0;
    calc_lighting(
        input.m_position_ws_local,
        normal_ws,
        roughness,
        diffuse_reflectance,
        specular_f0,
        input.m_planet_normal_ws,
        ao,
        ResourceDescriptorHeap[light_list_cbv],
        shadow_caster_count,
        shadow_caster_srv,
        exposure_value,
        dfg_texture_srv,
        diffuse_ld_texture_srv,
        specular_ld_texture_srv,
        dfg_texture_size,
        specular_ld_mip_count,
        gsm_srv,
        gsm_camera_view_local_proj,
        align_ground_rotation,
        direct_lighting,
        indirect_lighting);

    direct_lighting_result = float4(pack_lighting(direct_lighting), 0);
    indirect_lighting_result = float4(pack_lighting(indirect_lighting), 0);

    // When raytracing is used for diffuse GI - no need to compute indirect lighting
    // However, we need diffuse reflectance to multiply indirect lighting computed with DXR
    // Primary rays are not traced, so we need diffuse reflectance from V-Buffer pass
#if defined(MB_RAYTRACING_DIFFUSE_GI)
    indirect_lighting_result = float4(diffuse_reflectance, 0);
#endif

#if defined(MB_TILE_BORDER_ENABLED)
    draw_tile_border(input.m_position_local, direct_lighting_result, indirect_lighting_result);
#endif
}

ps_input_lighting_ocean vs_gpu_instancing(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t render_instance = render_instance_buffer[instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Get render item info
    StructuredBuffer<sb_render_item_t> render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t render_item = render_items_buffer[render_instance.m_render_item_idx];

    // Get material
    StructuredBuffer<sb_ocean_material> material_list = ResourceDescriptorHeap[NonUniformResourceIndex(render_item.m_material_buffer_srv)];
    sb_ocean_material material = material_list[render_item.m_material_index];

    // Get patch data
    StructuredBuffer<sb_ocean_tile_instance_t> tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(material.m_tile_instance_buffer_index)];

    // Unpack input data
    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    ps_input_lighting_ocean result = lighting_vertex_shader_ocean(
        vertex_id,
        render_item,
        material,
        render_instance,
        tile_instances,
        camera,
        instance_id,
        g_push_constants.m_time * time_scale);

    return result;
}

void ps_shadow_pass()
{
}

void ps_visibility_pass()
{
}

ps_output ps_main(ps_input_lighting_ocean input)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t render_instance = render_instance_buffer[input.m_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Get render item info
    StructuredBuffer<sb_render_item_t> render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t render_item = render_items_buffer[render_instance.m_render_item_idx];

    // Get material
    StructuredBuffer<sb_ocean_material> material_list = ResourceDescriptorHeap[NonUniformResourceIndex(render_item.m_material_buffer_srv)];
    sb_ocean_material material = material_list[render_item.m_material_index];

    // Get patch data
    StructuredBuffer<sb_ocean_tile_instance_t> tile_instances = ResourceDescriptorHeap[NonUniformResourceIndex(material.m_tile_instance_buffer_index)];
    sb_ocean_tile_instance_t tile = tile_instances[render_instance.m_user_data];

    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float4 direct_lighting = (float4)0;
    float4 indirect_lighting = (float4)0;
    lighting_pixel_shader_ocean(
        input,
        material,
        tile,
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
        (float3x3)camera.m_align_ground_rotation,
        g_push_constants.m_time * time_scale,
        direct_lighting,
        indirect_lighting);

    ps_output ps_output;
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    // Quadtree is static and won't move
    ps_output.m_velocity = 0;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    ps_output.m_entity_id = pack_entity_id(input.m_entity_id);
#endif

    ps_output.m_direct_lighting = direct_lighting;
    ps_output.m_indirect_lighting = indirect_lighting;

    return ps_output;
}
