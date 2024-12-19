// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MB_SHADER_SHADER_BUFFERS
#define MB_SHADER_SHADER_BUFFERS

#include "mb_shared_common.hlsl"

// This is HLSL shared with C implementation
// Use of only shared functionality is permitted!

//-----------------------------------------------------------------------------
// Define math types
//-----------------------------------------------------------------------------

#ifdef __cplusplus

// Define types to match with C++ code
#define double4 mb_math::double4
#define float3x3 mb_math::float3x3
#define float4x4 mb_math::float4x4
#define float4x3 mb_math::float4x3
#define float2 mb_math::float2
#define float3 mb_math::float3
#define float4 mb_math::float4
#define int2 mb_math::int2
#define int3 mb_math::int3
#define int4 mb_math::int4
#define uint2 mb_math::uint2
#define uint3 mb_math::uint3
#define uint4 mb_math::uint4
#define uint uint32_t
#define int int32_t

// Define macro for passing structures as push constants
#define RAL_PUSH_CONSTANTS(l_push_constants) sizeof(l_push_constants), &l_push_constants

// Constant buffer rules are strict and cannot be fully emulated on C++ side
// This results in CPU/GPU structure alignment to be different and unpredictable bugs
// The closes way to emulate is is to use pragma pack(4)
// For more details refer to https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
//! \todo DXC compiler currently does not support turning off cb packing, but it might in the future
#pragma pack(push, 4)

namespace mb_shader_buffers
{
#endif

//-----------------------------------------------------------------------------
// Constant buffers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct cb_push_gltf_t
{
    uint m_camera_cbv;
    uint m_render_item_buffer_srv;
    uint m_render_instance_buffer_srv;
    uint m_render_instance_buffer_offset;

    uint m_dfg_texture_srv;
    uint m_dfg_texture_size;
    uint m_diffuse_ld_texture_srv;
    uint m_specular_ld_texture_srv;

    uint m_specular_ld_mip_count;
    float m_exposure_value;
    uint m_light_list_cbv;
    uint m_time; // milliseconds

    uint m_shadow_caster_count;
    uint m_shadow_caster_srv;
    uint m_padding_0;
    uint m_gsm_srv;

    float4x4 m_gsm_camera_view_local_proj;
};

//-----------------------------------------------------------------------------
struct cb_camera_t
{
    float4x4 m_view_local;
    float4x4 m_proj;
    float4x4 m_view_proj_local_prev; // Stored in a current's frame camera local space
    float4x4 m_view_proj_local;
    float4x4 m_inv_view_proj_local;
    float4x4 m_align_ground_rotation;
    float4 m_frustum_planes[12]; // 6 frustum planes + up to 6 additional planes(for example during shadow pass)

    float4 m_fov_vertical; // (angle, tan(0.5f * angle), unused, unused)

    float3 m_camera_pos;

    float m_z_near;
    float m_z_far;

    // Fractional part of the camera computed as: m_camera_pos_frac_N = (float3)frac((double3)m_camera_pos / N)
    float3 m_camera_pos_frac_1;

    float3 m_camera_pos_frac_100;
    float m_resolution_x;

    float3 m_camera_pos_frac_10000;
    float m_resolution_y;

    float2 m_render_scale;

    uint m_num_frustum_planes;

    uint m_render_output_mask;
};

//-----------------------------------------------------------------------------
struct cb_push_adapted_luminance_t
{
    uint m_adapted_lum_texture_srv;
    uint m_lum_texture_srv;

    float m_prev_adapted_lum_uvx;
    float m_delta_time;

    float m_speed_up;
    float m_speed_down;
    float m_min_avg_lum;
    float m_max_avg_lum;
};

//-----------------------------------------------------------------------------
struct cb_push_luminance_t
{
    uint m_src_texture_srv;
    uint m_dst_texture_uav;
    uint2 m_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_plot_adapted_luminance_t
{
    uint2 m_adapted_lum_texture_srv;
    uint m_cur_adapted_lum_index;

    uint m_dst_resolution_y;

    float2 m_lum_bounds; // x: lower bound, y: upper bound
};

//-----------------------------------------------------------------------------
struct cb_push_debug_luminance_t
{
    uint m_src_texture_srv;
    float2 m_lum_bounds; // x: lower bound, y: upper bound
};

//-----------------------------------------------------------------------------
struct cb_push_tone_mapping_fixed_exposure_t
{
    uint m_hdr_texture_srv;
    float m_key_value;
    uint m_tonemap_operator;
};

//-----------------------------------------------------------------------------
struct cb_push_exposure_fusion_exposures_t
{
    uint m_hdr_texture_srv;
    uint m_lum_texture_srv;
    float m_lum_uvx;
    float m_key_value;
    float m_exposure_highlights;
    float m_exposure_shadows;
};

//-----------------------------------------------------------------------------
struct cb_push_exposure_fusion_weights_t
{
    uint m_texture_srv;
    float m_sigma_squared;
};

//-----------------------------------------------------------------------------
struct cb_push_exposure_fusion_blend_t
{
    uint m_weights_texture_srv;
    uint m_exposures_texture_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_exposure_fusion_blend_laplacian_t
{
    uint m_exposures_texture_srv;
    uint m_exposures_coarser_texture_srv;
    uint m_boost_local_constrast;
    uint m_weights_texture_srv;
    uint m_accum_texture_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_exposure_fusion_final_combine_t
{
    uint m_hdr_texture_srv;
    uint m_blend_texture_srv;
    uint m_exposures_mip_texture_srv;
    uint m_lum_texture_srv;
    float2 m_inv_pixel_size;
    float m_lum_uvx;
    float m_key_value;
};

//-----------------------------------------------------------------------------
struct cb_push_tone_mapping_eye_adaptation_t
{
    uint m_hdr_texture_srv;
    float m_key_value;
    uint m_tonemap_operator;
    uint m_lum_texture_srv;
    float m_lum_uvx;
};

//-----------------------------------------------------------------------------
struct cb_push_hdr_off_t
{
    uint m_src_texture_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_upscaling_t
{
    uint m_camera_cbv;
    uint m_src_texture_srv;
    uint m_dst_texture_uav;

    uint m_padding;

    uint2 m_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_debug_cubemap_flat_view_t
{
    uint m_cubemap_texture_srv;
    uint m_face_id;
};

//-----------------------------------------------------------------------------
struct cb_push_spot_meter_t
{
    uint m_src_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_copy_t
{
    uint m_src_texture_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_memset_uav
{
    float4 m_value;
    uint m_texture_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_downsample_2x2_t
{
    uint m_src_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_precomputed_step;

    uint2 m_dst_resolution;
    float2 m_inv_src_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_test_compute_t
{
    uint m_rt_index;
};

//-----------------------------------------------------------------------------
struct cb_light_t
{
    uint   m_type;
    float3 m_color;

    float4 m_param_0;
    float4 m_param_1;
    float4 m_param_2;
};
struct cb_light_list_t
{
    cb_light_t m_light_list[MB_MAX_LIGHTS];
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_t
{
    uint m_raytracing_accumulation_rt_uav;
    uint m_raytracing_acc_struct_srv;
    uint m_camera_cbv;
    uint m_render_item_buffer_srv;

    uint m_rand;
    uint m_acc_frame_index;
    uint m_num_bounces;
    uint m_num_samples;

    uint m_cubemap_texture_srv;
    uint m_indirect_diffuse;
    uint m_indirect_specular;
    uint m_direct_lighting;

    uint m_environment_lighting;
    uint m_emissive_lighting;
    uint m_atmospheric_scattering_enabled;
    uint m_atmospheric_scattering_srv;

    float3 m_world_origin_local;
    float m_exposure_value;

    uint m_tile_hierarchy_srv;
    uint m_raytracing_normals_rt_uav;
    uint m_generate_new_frame;
    uint m_light_list_cbv;

    uint m_time; // milliseconds
    uint m_raytracing_accumulation_rt_srv;
    uint m_vt_feedback_buffer_capacity;
    uint m_depth_texture_srv;

    float2 m_inv_dst_resolution;
    uint m_raytracing_resolution_scale;
    uint m_velocity_texture_srv;

    uint m_raytracing_frame_count_texture_uav;
    uint m_raytracing_frame_count_texture_srv;
    uint m_raytracing_normals_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_reconstruct_normal_t
{
    uint m_camera_cbv;
    uint m_depth_texture_srv;
    uint m_dst_normal_texture_uav;

    uint m_padding;

    uint2 m_dst_resolution;
    float2 m_inv_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_upsampling_t
{
    uint m_raytracing_accumulation_rt_srv;
    uint m_direct_lighting_srv;
    uint m_diffuse_reflectance_srv;
    uint m_depth_texture_srv;

    uint m_output_uav;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_raytracing_resolution_scale;

    uint m_acc_frame_index;
    uint m_ssao_rt_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_denoising_t
{
    uint m_raytracing_accumulation_rt_srv;
    uint m_raytracing_accumulation_rt_uav;
    uint m_depth_texture_srv;
    uint m_dst_resolution_x;

    uint m_dst_resolution_y;
    uint m_denoising_iteration;
    uint m_raytracing_frame_count_texture_srv;
    uint m_camera_cbv;

    float m_poisson_kernel_rot_cos;
    float m_poisson_kernel_rot_sin;
    uint m_raytracing_normals_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_denoising_wavelet_t
{
    uint m_texture_dst_specular_uav;
    uint m_texture_dst_uav;
    uint m_indirect_specular_uav;
    uint m_indirect_diffuse_uav;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_acc_frame_index;
    uint m_num_frames;
    uint m_frame_index_since_history_reset;
    uint m_raytracing_depth_rt_uav;
    uint m_raytracing_normals_rt_uav;
    uint m_step_size;
};

//-----------------------------------------------------------------------------
struct cb_push_raytracing_combine_t
{
    uint m_texture_dst_uav;
    uint m_accumulated_lighting_uav;
    uint m_direct_lighting_uav;
    uint m_indirect_diffuse_uav;
    uint m_indirect_specular_uav;
    uint m_denoised_indirect_diffuse_uav;
    uint m_denoised_indirect_specular_uav;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_acc_frame_index;
    uint m_raytracing_diffuse_reflectance_rt_uav;
    uint m_raytracing_depth_rt_uav;
    uint m_raytracing_normals_rt_uav;
    uint m_output_type;
};

//-----------------------------------------------------------------------------
struct cb_push_background_t
{
    uint m_camera_cbv;
    uint m_cubemap_texture_srv;
    float m_exposure_value;
};

//-----------------------------------------------------------------------------
struct cb_push_tile_generation_t
{
    // Output textures
    uint m_tile_heightmap_index;
    uint m_tile_texturemap_index;
    uint m_tile_layer_mask_0_index;
    uint m_tile_layer_mask_1_index;
    uint m_tile_layer_mask_2_index;
    uint m_tile_layer_mask_3_index;
    uint m_tile_texture_slice;          // 7 dwords

    // Elevation
    uint m_elevation_buffer_srv;
    uint m_elevation_data_offset;
    uint2 m_elevation_resolution;

    uint m_texture_channel_offset;
    uint m_texture_buffer_srv;
    uint m_texture_data_offset;
    uint2 m_texture_resolution;         // 16 dwords

    // Splat
    uint2 m_splat_resolution;
    uint m_splat_buffer_srv;
    uint m_splat_data_offset;
    uint m_splat_channel_offset;

    uint m_splat_channels;              // 22 dwords
};

//-----------------------------------------------------------------------------
struct cb_push_tile_generation_population_t
{
    // Output textures
    uint m_tile_heighmap_srv;
    uint m_tile_layer_mask_0_srv;
    uint m_tile_layer_mask_1_srv;

    uint m_population_buffer_uav;

    // Elevation
    uint m_elevation_buffer_srv;
    uint2 m_elevation_resolution;
    uint m_elevation_border_width;

    // Splat
    uint m_splat_buffer_srv;

    // Population
    uint m_population_id_buffer_srv;
    uint m_population_id_offset;
    uint m_population_x_offset;
    uint m_population_y_offset;
    uint m_population_z_offset;
    uint m_population_scale_offset;
    uint m_population_orientation_offset;
    uint2 m_population_resolution;

    uint m_tile_texture_slice;

    uint m_tile_level;
    float m_tile_x;
    float m_tile_y;
    float m_tile_size;
};

//-----------------------------------------------------------------------------
struct cb_push_tile_population_update_t
{
    uint m_population_update_buffer_srv;
    uint m_update_count;
    uint m_population_buffer_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_compact_population_update_t
{
    uint m_population_update_buffer_srv;
    uint m_update_count;
    uint m_compact_population_count_srv;
    uint m_compact_population_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_vt_generation_t
{
    // Output textures
    uint m_tile_heightmap_index;
    uint m_tile_layer_mask_0_index;
    uint m_tile_layer_mask_1_index;
    uint m_tile_layer_mask_2_index;
    uint m_tile_layer_mask_3_index;     // 5 dwords

    // Virtual texture tile
    uint m_tile_vt0_tmp_uav;
    uint m_tile_vt1_tmp_uav;            // 7 dwords

    uint m_tile_texture_slice;
    uint m_elevation_tile_border;
    uint2 m_elevation_tile_resolution;
    uint m_splat_tile_border;
    uint2 m_splat_tile_resolution;
    uint m_vt_border;
    uint m_vt_resolution;               // 16 dwords

    uint m_material_buffer_srv;
    uint m_material_count;
    float m_tile_x;
    float m_tile_y;
    float m_tile_size;
    float m_hex_tile_x;
    float m_hex_tile_y;
    float m_hex_tile_size;
    float m_mip_level;                  // 25 dwords
};

//-----------------------------------------------------------------------------
struct cb_push_tile_mesh_baking_t
{
    uint m_tile_position_buffer_uav;
    uint m_tile_normal_buffer_uav;
    uint m_tile_undisplaced_position_buffer_uav;
    uint m_tile_instance_buffer_srv;

    uint m_tile_heightmap_index;
    uint2 m_elevation_tile_resolution;
    uint m_elevation_tile_border;

    uint m_tile_vertex_resolution;
    uint m_num_vertices;

    float m_horizontal_displacement_scale;
};

//-----------------------------------------------------------------------------
struct cb_push_tile_population_t
{
    uint m_population_render_item_buffer_srv;
    uint m_instance_buffer_uav;
    uint m_instance_count_buffer_uav;
    uint m_instance_buffer_capacity;

    uint m_population_model_buffer_srv;
    uint m_tile_buffer_srv;
    uint m_tile_height_array_index_srv;
    uint m_elevation_tile_border;

    uint2 m_elevation_tile_resolution;
    uint m_compact_population_buffer_srv;
    uint m_compact_population_count_buffer_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_generate_dfg_t
{
    uint m_dst_texture_uav;
    uint2 m_dst_resolution;

    uint m_num_samples;
};

//-----------------------------------------------------------------------------
struct cb_push_generate_diffuse_ld_t
{
    uint m_camera_cbv;

    uint m_texture_srv;
    uint m_num_samples;
    uint m_mip_level;
};

//-----------------------------------------------------------------------------
struct cb_push_generate_specular_ld_t
{
    uint m_camera_cbv;

    uint m_texture_srv;
    uint m_level;
    uint m_mip_count;

    float m_inv_omega_p;
};

//-----------------------------------------------------------------------------
struct cb_camera_light_probe_t
{
    float4x4 m_inv_view_proj_local;
    float4x4 m_align_ground_rotation;
};

//-----------------------------------------------------------------------------
struct cb_push_atmospheric_scattering_t
{
    uint m_camera_cbv;
    uint m_depth_texture_srv;
    uint m_dst_texture_uav;
    uint m_lut_texture_srv;                     // 4 dwords

    // Solid parameters
    float3 m_planet_center;
    float m_planet_radius;                      // 8 dwords

    uint2 m_dst_resolution;

    float2 m_density_scale_height;
    float m_atmosphere_height;                  // 13 dwords

    // Tunable parameters
    float3 m_rayleigh_scattering_coefficient;
    float3 m_mie_scattering_coefficient;
    float m_mie_g;
    float3 m_solar_irradiance;                  // 23 dwords

    // Parameters related to samples
    uint m_sample_count_light_direction;
    uint m_sample_count_view_direction;         // 25 dwords

    // Directional light source
    float3 m_sun_light_dir;
    float3 m_sun_light_color;                   // 31 dwords

    // Stars
    float m_star_intensity;

    // Debugging
    uint m_enable_rayleigh_scattering;
    uint m_enable_mie_scattering;               // 34 dwords
};

//-----------------------------------------------------------------------------
struct cb_push_generate_scattering_lut_t
{
    uint m_dst_texture_uav;
    uint2 m_dst_resolution;

    float m_planet_radius;
    float m_atmosphere_height;
    float2 m_density_scale_height;

    uint m_sample_count;
};

//-----------------------------------------------------------------------------
struct cb_push_instancing_t
{
    uint m_command_buffer_uav;
    uint m_command_buffer_srv;
    uint m_instance_buffer_srv;
    uint m_instance_count_buffer_srv;

    uint m_render_item_buffer_srv;
    uint m_render_item_count;
    uint m_scratch_buffer_index;
    uint m_instance_buffer_final_index;

    uint m_lod_camera_cbv;
    float m_lod_camera_offset_x;
    float m_lod_camera_offset_y;
    float m_lod_camera_offset_z;

    uint m_hiz_map_srv;
    float m_impostor_distance;
    float m_lod_bias;
    uint m_padding_1;

    cb_push_gltf_t m_push_constants_gltf;
};

//-----------------------------------------------------------------------------
struct cb_procedural_splat_t
{
    uint m_tile_level;
    uint m_tile_offset;
    uint m_splat_offset;
    uint m_splat_channels;

    uint2 m_resolution;
    uint m_elevation_buffer_srv;
    uint m_elevation_offset;

    uint m_elevation_resolution_x;
    uint m_elevation_resolution_y;
    uint m_elevation_border_width;
    uint m_border_width;

    // Computed on cube, (pos_x, pos_y, period_xy)
    // Usage: float2 l_pos = frac(m_tile_pos_frac_100.x + l_uv * m_tile_pos_frac_100.z);
    float3 m_tile_pos_frac_100;
    uint m_tree_index;
    
    float3 m_tile_pos_frac_10000;
    uint m_buffer_uav;
    
    float3 m_tile_pos_frac_1000000;
    float m_tile_ao_radius_meters;

    float m_tile_ao_radius_uv;
    float m_tile_x;
    float m_tile_y;
    float m_tile_size;

    uint m_tile_index;
    uint m_mask_buffer_srv;
    uint m_mask_resolution;
    uint m_mask_channel_count;
};

//-----------------------------------------------------------------------------
struct cb_procedural_population_t
{
    uint m_population_buffer_uav;
    uint m_population_dimension;

    uint m_tile_seed;
    uint m_tile_level;
    float m_tile_x;
    float m_tile_y;
    float m_tile_size;
    uint m_tree_index;

    uint m_elevation_buffer_srv;
    uint m_elevation_offset;
    uint m_elevation_resolution_x;
    uint m_elevation_resolution_y;
    uint m_elevation_border_width;

    uint m_tile_index;
    uint m_mask_buffer_srv;
    uint m_mask_resolution;
    uint m_mask_channel_count;
};

//-----------------------------------------------------------------------------
struct cb_push_population_compact_init_t
{
    uint m_compact_population_count_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_population_compact_copy_t
{
    uint m_src_compact_population_count_srv;
    uint m_src_compact_population_srv;

    uint m_dst_compact_population_count_uav;
    uint m_dst_compact_population_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_population_compact_reset_t
{
    uint m_tile_index;

    uint m_compact_population_count_srv;
    uint m_compact_population_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_population_compact_append_t
{
    uint m_population_buffer_srv;
    uint m_tile_index;
    uint m_compact_population_count_uav;
    uint m_compact_population_uav;

    uint m_compact_population_capacity;
};

//-----------------------------------------------------------------------------
struct cb_compress_bc1_t
{
    uint m_texture_src_srv;
    uint m_texture_dst_uav;
    uint m_input_is_srgb;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_src_resolution_x;
    uint m_src_resolution_y;
};

//-----------------------------------------------------------------------------
struct cb_compress_bc3_t
{
    uint m_texture_src_srv;
    uint m_texture_dst_uav;
    uint m_input_is_srgb;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
};

//-----------------------------------------------------------------------------
struct cb_compress_bc5_t
{
    uint m_texture_src_srv;
    uint m_texture_dst_uav;
    uint m_input_is_srgb;
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_src_resolution_x;
    uint m_src_resolution_y;
};

//-----------------------------------------------------------------------------
struct cb_push_smaa_edge_detection_t
{
    uint m_color_texture_srv;
    uint m_depth_texture_srv;
    float m_render_target_width;
    float m_render_target_height;
};

//-----------------------------------------------------------------------------
struct cb_push_smaa_blend_weights_t
{
    uint m_edges_texture_srv;
    uint m_area_texture_srv;
    uint m_search_texture_srv;
    float m_render_target_width;
    float m_render_target_height;
};

//-----------------------------------------------------------------------------
struct cb_push_smaa_blend_t
{
    uint m_color_texture_srv;
    uint m_blend_texture_srv;
    float m_render_target_width;
    float m_render_target_height;
};

//-----------------------------------------------------------------------------
struct cb_push_debug_rendering_t
{
    float4x4 m_view_proj_matrix;
    uint m_vertex_buffer_srv;
};

struct cb_push_debug_rendering_indirect_t
{
    float4x4 m_view_proj_matrix;            //92  bytes

    uint m_padding;

    uint m_vertex_buffer_srv;               //96  bytes
    uint m_draw_command_buffer_uav;         //100 bytes
    uint m_counter_resource_srv;            //104 bytes
};

struct cb_push_clear_counter_t
{
    uint m_counter_resource_uav;
};

//-----------------------------------------------------------------------------
struct cb_test_values_t
{
    float4 m_float_val;
    int4 m_int_val;
};

//-----------------------------------------------------------------------------
struct cb_vt_test_t
{
    uint m_src_texture_srv;
    uint m_src_texture_sampler_feedback_uav;
    uint m_src_texture_residency_map_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_debug_visualize_texture_t
{
    uint m_src_texture_srv;
    uint2 m_src_texture_size;
    uint m_lowest_value_texture_srv;
    uint m_highest_value_texture_srv;

    uint m_is_uint_format;
    uint m_format_num_channels;
    uint m_is_texture_array;
    uint m_texture_array_index;

    uint3 m_channel_swtiches; // 0: disabled, 1: enabled
    uint m_texture_mip_index;

    float3 m_lower_bounds; // 0 by default
    float3 m_upper_bounds; // 1 by default

    uint m_is_hdr_color;
};

//-----------------------------------------------------------------------------
struct cb_push_downsample_with_comparison_t
{
    uint m_src_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_dst_resolution;
    float2 m_inv_src_resolution;

    uint2 m_precomputed_step;
};

//-----------------------------------------------------------------------------
struct cb_push_reconstruct_normal_t
{
    uint m_camera_cbv;
    uint m_depth_texture_srv;
    uint m_dst_normal_texture_uav;

    uint m_padding;

    uint2 m_dst_resolution;
    float2 m_inv_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_ssao_kernel_t
{
    float3 m_samples[64];
};

//-----------------------------------------------------------------------------
struct cb_push_ssao_t
{
    uint m_camera_cbv;
    uint m_kernel_cbv;
    uint m_depth_texture_srv;
    uint m_normal_texture_srv;
    uint m_noise_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_dst_resolution;
    float2 m_inv_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_gtao_t
{
    uint m_camera_cbv;
    uint m_depth_texture_srv;
    uint m_normal_texture_srv;
    uint m_jitter_texture_srv;
    uint m_dst_texture_uav;

    uint m_padding;

    uint2 m_dst_resolution;
    float2 m_inv_dst_resolution;

    float m_sample_aspect_ratio;

    float m_far_fade_out_threshold;
    float m_far_fade_out_range;
    float m_near_fade_in_begin;
    float m_near_fade_in_end;

    float m_small_scale_ao_amount;
    float m_large_scale_ao_amount;
    float m_intensity;
    float m_half_project_scale;

    float m_near_radius;
    float m_far_radius;
    float m_near_horizon_falloff;
    float m_far_horizon_falloff;

    float m_near_thickness;
    float m_far_thickness;
    float m_fade_start;
    float m_fade_speed;
};

//-----------------------------------------------------------------------------
struct cb_push_gtao_blur_t
{
    uint m_camera_cbv;
    uint m_depth_texture_srv;
    uint m_ao_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_dst_resolution;
    float2 m_inv_dst_resolution;

    float m_far_fade_out_threshold;

    float m_power_exponent;
    float m_blur_sharpness;
};

//-----------------------------------------------------------------------------
struct cb_push_gtao_downsample_t
{
    uint m_src_texture_srv;
    uint m_dst_texture_uav;

    uint2 m_dst_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_lighting_combination_t
{
    uint m_camera_cbv;
    uint m_direct_lighting_rt_srv;
    uint m_indirect_lighting_rt_srv;
    uint m_ssao_rt_srv;

    uint m_scene_rt_uav;
    uint2 m_dst_resolution;
};

struct cb_push_deferred_lighting_t
{
    uint m_camera_cbv;
    uint m_visibility_buffer_srv;
    uint m_render_item_buffer_srv;
    uint m_render_instance_buffer_srv;
    uint m_direct_lighting_uav;
    uint m_indirect_lighting_uav;
    uint m_velocity_uav;

    uint m_time;

    float2 m_dst_resolution;

    uint m_light_list_cbv;
    uint m_shadow_caster_count;
    uint m_shadow_caster_srv;
    float m_exposure_value;
    uint m_dfg_texture_srv;
    uint m_diffuse_ld_texture_srv;
    uint m_specular_ld_texture_srv;
    uint m_dfg_texture_size;
    uint m_specular_ld_mip_count;

    uint m_lighting_classification_texture_srv;
    uint m_lighting_classification_type;

    uint2 m_padding;
    uint m_gsm_srv;
    float4x4 m_gsm_camera_view_local_proj;
};

//-----------------------------------------------------------------------------
struct cb_push_vt_feedback_apply_t
{
    uint m_vt_feedback_buffer_capacity;
    uint m_render_item_buffer_srv;
};

//-----------------------------------------------------------------------------
struct cb_push_motion_blur_t
{
    uint m_dst_resolution_x;
    uint m_dst_resolution_y;
    uint m_src_texture_srv;
    uint m_velocity_texture_srv;
    uint m_dst_texture_uav;
    uint m_depth_texture_srv;
    uint m_camera_cbv;
    float m_kernel_size_multiplier;
    float m_object_threshold_velocity;
    float m_object_max_velocity;
    float m_object_velocity_scale;
    float m_camera_threshold_velocity;
    float m_camera_max_velocity;
    float m_camera_velocity_scale;
    uint m_pass_index;
};

//-----------------------------------------------------------------------------
struct cb_push_impostor_t
{
    uint m_camera_cbv;
    uint m_dfg_texture_srv;
    uint m_dfg_texture_size;
    uint m_diffuse_ld_texture_srv;

    uint m_specular_ld_texture_srv;
    uint m_specular_ld_mip_count;
    float m_exposure_value;
    uint m_light_list_cbv;

    uint m_item_buffer_srv;
    uint m_item_count;
    uint m_instance_buffer_srv;
    uint m_instance_count_buffer_srv;

    uint m_start_instance_location;
    uint m_command_buffer_uav;
    uint m_item_count_buffer_uav;
    uint m_sorted_instance_buffer_uav;

    uint m_sorted_instance_buffer_srv;
    uint m_hiz_map_srv;
    uint m_gsm_srv;
    float m_start_distance;

    float4x4 m_gsm_camera_view_local_proj;
};

//-----------------------------------------------------------------------------
struct cb_push_tile_impostor_population_t
{
    uint m_compact_population_buffer_srv;
    uint m_compact_population_count_buffer_srv;
    uint m_instance_buffer_uav;
    uint m_instance_count_buffer_uav;

    uint m_instance_buffer_capacity;
    uint m_tile_buffer_srv;
    uint m_tile_height_array_index_srv;
    uint m_elevation_tile_border;

    uint2 m_elevation_tile_resolution;
};

//-----------------------------------------------------------------------------
struct cb_push_tile_impostor_population_debug_t
{
    float3 m_position;
    float m_angle;
    float3 m_up_vector;
    float m_scale;
    uint m_item_idx;

    uint m_instance_buffer_capacity;
    uint m_instance_count_buffer_uav;
    uint m_instance_buffer_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_hiz_map
{
    uint m_src_srv;
    uint m_dst_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_hiz_pre_pass
{
    float3 m_camera_diff;
    uint m_camera_cbv;
    uint m_command_buffer_uav;
    uint m_instance_buffer_final_uav;
    uint m_quadtree_instance_buffer_default_uav;
    uint m_time;
};

//-----------------------------------------------------------------------------
struct cb_push_moon_t
{
    uint m_camera_cbv;
    uint m_albedo_texture_srv;
    uint m_normal_texture_srv;      // 3 dwords

    float m_edge_transition_width;
    float m_edge_transition_radius;

    // Used to scale the moon lighting
    float3 m_k_lambda;              // 8 dwords

    float3 m_moon_position;
    float m_billboard_size;

    float3 m_light_dir;
    float m_lunar_phase_angle;      // 16 dwords

    float4x4 m_proj;                // 32 dwords
};

//-----------------------------------------------------------------------------
struct cb_push_resolve_gsm_t
{
    uint2 m_gsm_dimensions;

    uint m_gsm_srv;
    uint m_resolved_texture_uav;

    float3 m_march_dir;
    float m_gsm_z_near;
    float m_gsm_z_far;

    uint2 m_mouse_pos;          // Debug only
    uint m_debug_texture_uav;   // Debug only
};

//-----------------------------------------------------------------------------
struct cb_push_grass_patch_preparation_t
{
    uint m_preparation_buffer_srv;
    uint m_tile_position_buffer_srv;
    uint m_tile_normal_buffer_srv;
    uint m_tile_vt0_array_index_srv;
    uint m_patch_item_buffer_uav;

    uint m_tile_vertex_resolution;

    uint m_num_patches;

    // Mask
    uint m_mask_buffer_srv;
    uint m_mask_channel_count;
    uint m_mask_resolution;
    uint m_elevation_resolution;
    uint m_elevation_border_width;
};

//-----------------------------------------------------------------------------
struct cb_push_grass_culling_lod_t
{
    uint m_camera_cbv;
    uint m_patch_instance_buffer_srv;
    uint m_patch_tile_to_camera_offset_buffer_srv;
    uint m_patch_item_buffer_srv;
    uint m_patch_lod_count_buffer_uav;

    // This is supposed to be a UAV array with a size equal to LOD_LEVEL_COUNT (see mb_render_grass.hpp).
    // In HLSL, each element in an array aligns to a 16-byte boundary, even though a uint is only 4 bytes.
    // To aviod wasting memory, we use vector(s) instead of a uint array here.
    uint2 m_patch_arguments_uavs;

    uint m_patch_instance_buffer_offset;
    uint m_patch_arguments_capacity;
    float m_culling_distance_squared;
};

//-----------------------------------------------------------------------------
struct cb_push_grass_lod_reset_t
{
    uint m_num_lod_levels;
    uint m_count_buffer_uav;
};

//-----------------------------------------------------------------------------
struct cb_push_grass_t
{
    uint m_camera_cbv;
    uint m_patch_lod_count_buffer_srv;
    uint m_patch_arguments_srv;

    uint m_lod_level;
    float m_ground_blend_start_distance;
    float m_ground_blend_range_reciprocal;

    uint m_dfg_texture_srv;
    uint m_dfg_texture_size;
    uint m_diffuse_ld_texture_srv;
    uint m_specular_ld_texture_srv;
    uint m_specular_ld_mip_count;
    float m_exposure_value;

    float4x4 m_gsm_camera_view_local_proj;
    uint m_gsm_srv;

    uint m_light_list_cbv;
    uint m_shadow_caster_count;
    uint m_shadow_caster_srv;

    // Wind
    uint m_time;
    float m_wind_scale;
    float m_wind_direction_rad;
};

struct cb_push_crash_t
{
    uint m_loop_count;
    uint m_buffer_uav;
    uint m_invalid_uav_index;
};

//-----------------------------------------------------------------------------
// Structured buffers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct sb_debug_shape_material_t
{
    uint m_type;
    uint m_sphere_size_in_quads;
    uint m_instance_data_srv;
};

//-----------------------------------------------------------------------------
struct sb_debug_shape_instance_data_t
{
    float4 m_color;
};

//-----------------------------------------------------------------------------
struct sb_quadtree_material_t
{
    uint m_tile_instance_buffer_index;
    uint m_tile_border_enabled;
    uint m_tile_height_array_index_srv;
    uint m_tile_texture_array_index_srv;
    uint m_tile_vt0_array_index_srv;
    uint m_tile_vt1_array_index_srv;
    uint2 m_elevation_tile_resolution;
    uint m_elevation_tile_border;

    uint2 m_texture_tile_resolution;
    uint m_texture_tile_border;

    uint m_vt_resolution;
    uint m_vt_border;
    float m_blend_range;

    uint m_ml_texture_only;
    uint m_splatmap_only;
    float m_ml_texture_blend_distance;
    float m_ml_texture_blend_strength;

    uint m_tile_size_in_vertices; // Tile mesh is a square, so both dimension match

    float3 m_height_debug_start_color;
    float m_height_debug_start;
    float3 m_height_debug_end_color;
    float m_height_debug_end;
    uint m_debug_terrain_gradient_scale;

    uint m_debug_terrain_mode; // height, <not used>, splat map, virtual textures, basemap debug, terrain
    uint m_debug_sub_mode;
    uint m_visualization_type; // float, unorm, snorm, id
    uint m_debug_terrain_splat_channel;

    uint m_tensor_debug_buffer_srv;
    uint m_tensor_debug_channel_count;
    uint m_tensor_debug_buffer_resolution;
    uint m_tensor_debug_channel_mask;
    float m_tensor_visualizer_scalar;
    float3 m_tensor_visualizer_channel_color[3];

    uint m_debug_texture_for_ints;

    uint m_num_splat_map_channels_to_show;
    uint m_splat_map_channels_to_show[8]; // 0 to 15
    float3 m_splat_map_channel_colors[8];

    uint m_splat_channel_offset;
    uint4 m_splat_shape;
    
    float m_skirt_distance_threshold_squared; // Squared distance from tile vertex to camera (meters). We only apply skirt when it's close enough.
    float m_skirt_scale;
};

//-----------------------------------------------------------------------------
// Quadtree tile is approximated with a patch
// Control points are derived from paper: http://cs.engr.uky.edu/~cheng/PUBL/Paper_Nagata.pdf
struct sb_tile_instance_t
{
    uint m_available; // 0 - not available, 1 - ready for use

    // Quadtree index [0..5]
    uint m_quadtree_index;

    // Patch normals
    float3 m_normal_0;
    float3 m_normal_1;
    float3 m_normal_2;
    float3 m_normal_3;

    // Patch control points
    float3 m_patch_data_c00;
    float3 m_patch_data_c10;
    float3 m_patch_data_c01;
    float3 m_patch_data_c11;
    float3 m_patch_data_c20;
    float3 m_patch_data_c02;
    float3 m_patch_data_c12;
    float3 m_patch_data_c21;

    // Entity ID used by selection pass
    uint   m_entity_id;

    // Tile index and parent index in the buffer array
    uint   m_tile_index;
    uint   m_parent_index;
    uint   m_tile_level;
    uint2  m_tile_id_xy;

    // Estimated tile size used for normal generation
    float  m_tile_size_estimate;

    // Blend factor to blend between tile levels
    float  m_blend_to_parent;

    // UVs in parent's coordinate space
    float2 m_parent_uv_offset;

    //
    float4 m_neighbours;            // left, right, bottom, top
    float4 m_neighbours_diagonal;   // top-left, top-right, bottom-left, bottom-right

    // Splat SRV and offset, debug only
    uint m_splat_buffer_srv;
    uint m_splat_data_offset;

    // Offset from tile local space to camera local space
    float3 m_tile_local_to_camera_local;
};

//-----------------------------------------------------------------------------
// Approximately a subset of sb_tile_instance_t, used for tile mesh baking
struct sb_tile_instance_to_bake_t
{
    // Patch normals
    float3 m_normal_0;
    float3 m_normal_1;
    float3 m_normal_2;
    float3 m_normal_3;

    // Patch control pointes
    float3 m_patch_data_c00;
    float3 m_patch_data_c10;
    float3 m_patch_data_c01;
    float3 m_patch_data_c11;
    float3 m_patch_data_c20;
    float3 m_patch_data_c02;
    float3 m_patch_data_c12;
    float3 m_patch_data_c21;

    // Computed on cube, (pos_x, pos_y, period_xy)
    // Usage: float2 l_pos = frac(m_tile_pos_frac_100.x + l_uv * m_tile_pos_frac_100.z);
    float3 m_tile_pos_frac_10000;

    // Tile indices iin the buffer array
    uint m_tile_index;
    uint m_parent_index;
    uint m_parent_neighbour_x_index;
    uint m_parent_neighbour_y_index;

    // Neighbour tiles' relative positions of parent tile
    float3 m_parent_neighbour_x_offset;
    float3 m_parent_neighbour_y_offset;

    // Child index of current tile in parent tile's children
    uint m_child_index_in_parent;

    float m_horizontal_displacement_scale;
};

//-----------------------------------------------------------------------------
struct sb_grass_patch_preparation_t
{
    float2 m_uv;
    uint m_tile_index;
    uint m_random_seed;
    float m_blade_width;
    float m_blade_height;
    float m_patch_radius;
};

//-----------------------------------------------------------------------------
struct sb_grass_patch_item_t
{
    uint m_type_id;
    float3 m_position_tile_local;
    float3 m_ground_normal;
    float3 m_ground_color;
    float3 m_color;
    uint m_random_seed;
    float m_blade_width;
    float m_blade_height;
    float m_patch_radius;
};

//-----------------------------------------------------------------------------
struct sb_grass_patch_argument_t
{
    float3 m_position_camera_local;
    float3 m_ground_normal;
    float3 m_ground_color;
    float3 m_color;
    uint m_random_seed;
    float m_blade_width;
    float m_blade_height;
    float m_patch_radius;
    uint m_blade_count;
};

//-----------------------------------------------------------------------------
// Common render item description
struct sb_render_item_t
{
    // Index buffer
    uint m_index_buffer_srv;
    uint m_index_buffer_offset;
    uint m_index_buffer_stride;
    uint m_index_count;

    // TODO: repack together
    uint m_position_buffer_srv;
    uint m_position_offset;
    uint m_position_stride;
    uint m_normal_buffer_srv;
    uint m_normal_offset;
    uint m_normal_stride;
    uint m_uv0_buffer_srv;
    uint m_uv0_offset;
    uint m_uv0_stride;

    // Tangent is used in the raytracing as derivatives are not available
    uint m_tangent_buffer_srv;
    uint m_tangent_offset;
    uint m_tangent_stride;

    // Material
    uint m_material_buffer_srv;
    uint m_material_index;

    // Bounding volumes, in model space
    float4 m_bounding_sphere_ms; // (center_xyz, radius)
    float3 m_aabb_min_ms;
    float3 m_aabb_max_ms;

    // Screen-coverage threshold
    float2 m_screen_coverage_range;

    // Render item transform
    float4x3 m_transform;

    uint m_render_output_mask;

    uint m_lod_index; // ToDo this only need to be here in debug builds. However, shaders dont use MB_DEBUG
};

//-----------------------------------------------------------------------------
//! \brief Specifies render id list for each model. Later can be replaced with hierarchy
struct sb_raytracing_hierarchy_t
{
    uint m_render_item_id;
};

//-----------------------------------------------------------------------------
//! \brief For each tile we store the render item explicitly. No need for indirection
struct sb_raytracing_tile_hierarchy_t
{
    sb_render_item_t m_render_item;
};

//-----------------------------------------------------------------------------
struct sb_render_instance_population_t
{
    float3 m_position;
    float3 m_normal;
    float m_rotation;
    float m_scale;
    uint m_render_item_idx;         // Points to render item buffer SRV
    uint m_entity_id;               // Used for selection pass
};

//-----------------------------------------------------------------------------
struct sb_render_instance_t
{
    float4x3 m_transform;           // Transform
    float4x3 m_transform_prev;      // Transform of previous frame
    uint m_render_item_idx;         // Points to render item buffer SRV
    uint m_entity_id;               // Used for selection pass
    float m_custom_culling_scale;   // Max of (x, y, z) world scale(Used for culling)
    uint m_user_data;               // Custom user data that can be provided per instance
};

//-----------------------------------------------------------------------------
struct sb_impostor_item_t
{
    uint m_albedo_alpha_srv;
    uint m_normal_depth_srv;
    float m_octahedron_divisions;
    uint m_padding;
    float4 m_bounding_sphere;
};

//-----------------------------------------------------------------------------
struct sb_impostor_instance_t
{
    float3 m_position;
    float3 m_up_vector;
    float m_angle;
    float m_scale;
    uint m_item_idx;
};

//-----------------------------------------------------------------------------
// Geometry info buffer

//-----------------------------------------------------------------------------
struct sb_geometry_pbr_material_t
{
    float3  m_emissive_factor;

    // Base color
    float4  m_base_color_factor;
    uint    m_base_color_texture_srv;
    uint    m_base_color_residency_buffer_srv;
    uint    m_base_color_sampler_feedback_uav;
    uint2   m_base_color_residency_buffer_dim;

    // Base color
    uint m_alpha_mask_texture_srv;

    // Metallic-Roughness
    float   m_metallic_factor;
    float   m_roughness_factor;
    uint    m_metallic_roughness_texture_srv;
    uint    m_metallic_roughness_residency_buffer_srv;
    uint    m_metallic_roughness_sampler_feedback_uav;
    uint2   m_metallic_roughness_residency_buffer_dim;

    // Normal map
    uint    m_normal_map_texture_srv;
    uint    m_normal_map_residency_buffer_srv;
    uint    m_normal_map_sampler_feedback_uav;
    uint2   m_normal_map_residency_buffer_dim;

    // Occlusion(AO)
    uint    m_occlusion_texture_srv;
    uint    m_occlusion_residency_buffer_srv;
    uint    m_occlusion_sampler_feedback_uav;
    uint2   m_occlusion_residency_buffer_dim;

    // Emission
    uint    m_emission_texture_srv;

    // Alpha
    float   m_alpha_cutoff;
};

//-----------------------------------------------------------------------------
struct sb_shadow_caster_t
{
    uint m_shadow_camera_cbv;
    uint m_shadow_texture_srv;
    float m_shadow_camera_offset_x;
    float m_shadow_camera_offset_y;
    float m_shadow_camera_offset_z;
    float m_shadow_texel_size;
    float2 m_shadow_map_size;
    float2 m_inv_shadow_map_size;
    float m_shadow_constant_depth_bias;
    float m_max_shadow_casting_distance;
};

//-----------------------------------------------------------------------------
struct sb_population_item_t
{
    uint m_item_id;
    float3 m_offset;
    float m_scale;
    float m_rotation;
};

//-----------------------------------------------------------------------------
struct sb_population_tile_item_t
{
    uint m_tile_index;
    uint m_cell_index;
    sb_population_item_t m_population_item;
};

//-----------------------------------------------------------------------------
struct sb_debug_rendering_vertex_t
{
    float4 m_position;
    float4 m_color;
};

//-----------------------------------------------------------------------------
struct sb_debug_rendering_vertex_indirect_t
{
    float4 m_position;
    float4 m_color;
};

//-----------------------------------------------------------------------------
struct sb_population_update_t
{
    uint m_cell_index;
    uint m_tile_index;
    uint m_value;
};

//-----------------------------------------------------------------------------
struct sb_vt_feedback_t
{
    uint m_render_item_id;
    uint m_min_mip_level;
    float2 m_uv;
};

//-----------------------------------------------------------------------------
// Indirect draw

//-----------------------------------------------------------------------------
struct indirect_draw_arguments_t
{
    uint m_vertex_count_per_instance;
    uint m_instance_count;
    uint m_start_vertex_location;
    uint m_start_instance_location;
};

struct indirect_draw_indexed_arguments_t
{
    uint m_index_count_per_instance;
    uint m_instance_count;
    uint m_start_index_location;
    uint m_base_vertex_location;
    uint m_start_instance_location;
};

//-----------------------------------------------------------------------------
struct indirect_draw_instancing_t
{
    cb_push_gltf_t m_push_constants;
    indirect_draw_indexed_arguments_t m_draw;
};

//-----------------------------------------------------------------------------
struct indirect_draw_debug_rendering_t
{
    cb_push_debug_rendering_indirect_t m_push_constants;
    indirect_draw_arguments_t m_draw;
};

//-----------------------------------------------------------------------------
struct indirect_draw_impostor_t
{
    cb_push_impostor_t m_push_constants;
    indirect_draw_arguments_t m_draw;
};

//-----------------------------------------------------------------------------
// Payloads
//-----------------------------------------------------------------------------

// Support different types for storing random seed
//#define USE_RAND_XORSHIFT
#if defined(USE_RAND_XORSHIFT)
#define rand_type_t uint
#else
#define rand_type_t uint
#endif

//-----------------------------------------------------------------------------
struct pl_ray_payload_t
{
    uint        m_path_length;
    rand_type_t m_rand_state;
    float3      m_throughput;
    float3      m_radiance;     // Output
};

//-----------------------------------------------------------------------------
struct pl_shadow_ray_payload_t
{
    bool m_hit;
};

//-----------------------------------------------------------------------------
// Undefine math types

#ifdef __cplusplus
};

// Pop packing rules(see upper side of this file)
#pragma pack(pop)

// Undefine types to keep math types in a separate namespace
#undef double4
#undef float3x3
#undef float4x4
#undef float4x3
#undef float2
#undef float3
#undef float4
#undef int2
#undef int3
#undef int4
#undef uint2
#undef uint3
#undef uint4
#undef uint
#undef int
#endif // __cplusplus

#endif // MB_SHADER_SHADER_BUFFERS
