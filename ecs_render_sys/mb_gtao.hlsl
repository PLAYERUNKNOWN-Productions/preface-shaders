// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

#define FADE_DELTA 1e-4
#define INTERLEAVE_POSITION_VS_MASK 3

#if QUALITY_ULTRA
    #define NUM_DIRECTIONS 4
    #define NUM_STEPS 16
#elif QUALITY_HIGH
    #define NUM_DIRECTIONS 2
    #define NUM_STEPS 16
#elif QUALITY_MEDIUM
    // As the normal will affect the GTAO, only one sample will cause the ground be over occluded
    #define NUM_DIRECTIONS 2
    #define NUM_STEPS 8
#else
    #define NUM_DIRECTIONS 2
    #define NUM_STEPS 4
#endif

// Root constants
ConstantBuffer<cb_push_gtao_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 get_view_position(float2 p_uv, float p_z_near, float p_z_far, float2 p_tan_half_fov_xy)
{
    float l_depth = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_uv).r;
    float l_z_vs = get_view_depth_from_depth(l_depth, p_z_near, p_z_far);
    float3 l_position_vs = get_view_position(p_uv, l_z_vs, p_tan_half_fov_xy);

    return l_position_vs;
}

float3 get_view_normal(float2 p_uv)
{
    float3 l_normal_vs = bindless_tex2d_sample_level(g_push_constants.m_normal_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_uv).xyz;
    l_normal_vs = l_normal_vs * 2.0 - 1.0;
    return l_normal_vs;
}

float compute_distance_fade(float p_distance, float p_fade_start, float p_fade_speed)
{
    return saturate(max(0, p_distance - p_fade_start) * p_fade_speed);
}

float gtao_fast_sqrt(float p_x)
{
    // [Drobot2014a] Low Level Optimizations for GCN
    return asfloat(0x1FBD1DF5 + (asint(p_x) >> 1));
}

float2 gtao_fast_sqrt(float2 p_x)
{
    // [Drobot2014a] Low Level Optimizations for GCN
    return asfloat(0x1FBD1DF5 + (asint(p_x) >> 1));
}

float gtao_fast_acos(float p_x)
{
    // [Eberly2014] GPGPU Programming for Games and Science
    float l_res = -0.156583 * abs(p_x) + M_PI / 2.0;
    l_res *= gtao_fast_sqrt(1.0 - abs(p_x));
    return (p_x >= 0) ? l_res : (M_PI - l_res);
}

float2 gtao_fast_acos(float2 p_x)
{
    // [Eberly2014] GPGPU Programming for Games and Science
    float2 l_res = -0.156583 * abs(p_x) + M_PI / 2.0;
    l_res *= gtao_fast_sqrt(1.0 - abs(p_x));
    return select(p_x >= 0, l_res, M_PI - l_res);
}

float integrate_arc_cos_weight(float2 p_h, float p_n, float p_cos_n, float p_sin_n)
{
    float2 l_arc = -cos(2.0f * p_h - p_n) + p_cos_n + 2.0f * p_h * p_sin_n;
    return 0.25f * (l_arc.x + l_arc.y);
}

float compute_coarse_ao(float2 p_uv_position, float2 p_pixel_percent, float3 p_position_vs, float3 p_normal_vs,
                        float2 p_inv_resolution, float p_z_near, float p_z_far,
                        float2 p_tan_half_fov_xy, float p_sample_aspect_ratio, float p_near_radius, float p_far_radius,
                        float p_near_horizon_falloff, float p_far_horizon_falloff, float p_near_thickness,
                        float p_far_thickness, float p_fade_start, float p_fade_speed, float p_half_project_scale)
{
    // Divide by NUM_STEPS + 1, so that the farthest samples are not fully attenuated

    float2 l_inv_quater_resolution = p_inv_resolution;
    float3 l_fade_param = lerp(float3(p_near_radius, p_near_thickness, p_near_horizon_falloff),
                               float3(p_far_radius, p_far_thickness, p_far_horizon_falloff),
                               compute_distance_fade(p_position_vs.z, p_fade_start, p_fade_speed));

    const float l_radius = l_fade_param.x;
    const float l_thickness = l_fade_param.y;
    const float l_falloff_div_radius2 = l_fade_param.z / (l_radius * l_radius);
    const float l_step_size_pixels = max(min(l_radius * p_half_project_scale / p_position_vs.z, 512), NUM_STEPS) / (NUM_STEPS + 1);

    float3 l_bent_normal = 0;
    float l_occlusion = 0;
    float l_distance = length(p_position_vs);
    float3 l_direction_vs = -p_position_vs / l_distance;
    float l_bent_cone = 0;

    const float l_alpha = M_PI / NUM_DIRECTIONS;

    const float l_alpha_offset = l_alpha * p_pixel_percent.x;
    float l_total_weight = 0;
    [unroll]
    for (float l_direction_index = 0; l_direction_index < NUM_DIRECTIONS; l_direction_index++)
    {
        float l_angle = l_alpha * l_direction_index + l_alpha_offset;

        // Compute normalized 2D direction
        float3 l_slice_direction = float3(float2(cos(l_angle), sin(l_angle)), 0);
        float l_weight = length(l_slice_direction * float3(1, p_sample_aspect_ratio, 0));

        // Jitter starting sample within the first step
        float l_ray_pixels = (p_pixel_percent.y * l_step_size_pixels + 1.0f);

        float2 l_cos_horizon = -1.0f;
        float2 l_screen_direction = float2(l_slice_direction.x, -l_slice_direction.y);

        [loop]
        for (float l_step_index = 0; l_step_index < NUM_STEPS; l_step_index++)
        {
            float2 l_uv_offset = round(l_ray_pixels * l_screen_direction);
            float4 l_snapped_uv_position = p_uv_position.xyxy + float4(l_uv_offset.xy, -l_uv_offset.xy);
            float4 l_snapped_uv = l_snapped_uv_position * l_inv_quater_resolution.xyxy;

            float3 l_position_vs_1 = get_view_position(l_snapped_uv.xy, p_z_near, p_z_far, p_tan_half_fov_xy);
            float3 l_position_vs_2 = get_view_position(l_snapped_uv.zw, p_z_near, p_z_far, p_tan_half_fov_xy);

            float3 l_h1 = l_position_vs_1 - p_position_vs;
            float3 l_h2 = l_position_vs_2 - p_position_vs;

            float2 l_h1_h2 = float2(dot(l_h1, l_h1), dot(l_h2, l_h2));
            float2 l_h1_h2_length = rsqrt(l_h1_h2);

            float2 l_falloff = saturate(l_h1_h2 * l_falloff_div_radius2); // Close to r, -> stable

            float2 l_h = float2(dot(l_h1, l_direction_vs), dot(l_h2, l_direction_vs)) * l_h1_h2_length;
            l_cos_horizon.xy = lerp(l_h, l_cos_horizon, select(l_h.xy > l_cos_horizon.xy, l_falloff, l_thickness));
            l_ray_pixels += l_step_size_pixels;
        }

        float3 l_slice_normal = normalize(cross(l_slice_direction, l_direction_vs));
        float3 l_slice_tangent = normalize(cross(l_direction_vs, l_slice_normal));
        float3 l_view_normal_project_to_slice = p_normal_vs - l_slice_normal * dot(p_normal_vs, l_slice_normal);
        float l_vnps_length = length(l_view_normal_project_to_slice);
        l_view_normal_project_to_slice /= l_vnps_length;
        float l_cos_vnps_with_view = clamp(dot(l_view_normal_project_to_slice, l_direction_vs), -1.0f, 1.0f);
        float l_cos_vnps_with_tangent = dot(l_view_normal_project_to_slice, l_slice_tangent);
        float l_angle_vnps_with_view = gtao_fast_acos(l_cos_vnps_with_tangent) - M_PI * 0.5;
        float l_sin_vnps_with_view = -l_cos_vnps_with_tangent;

        l_cos_horizon = gtao_fast_acos(clamp(l_cos_horizon, -1.0f, 1.0f)) * float2(-1.0f, 1.0f);
        l_cos_horizon.x = l_angle_vnps_with_view + max(l_cos_horizon.x - l_angle_vnps_with_view, -M_PI * 0.5);
        l_cos_horizon.y = l_angle_vnps_with_view + min(l_cos_horizon.y - l_angle_vnps_with_view, M_PI * 0.5);
        l_bent_cone += abs(l_cos_horizon.x - l_cos_horizon.y);

        float l_bent_angle = (l_cos_horizon.x + l_cos_horizon.y) * 0.5;
        l_bent_normal += l_direction_vs * cos(l_bent_angle) - l_slice_tangent * sin(l_bent_angle); // Rotate a bent angle from view direction
        l_occlusion += l_vnps_length * integrate_arc_cos_weight(l_cos_horizon, l_angle_vnps_with_view, l_cos_vnps_with_view, l_sin_vnps_with_view) * l_weight;

        l_total_weight += l_weight;
    }

    l_bent_normal = normalize(normalize(l_bent_normal) - l_direction_vs * 0.5f);
    l_occlusion /= l_total_weight;
    l_bent_cone /= NUM_DIRECTIONS;

    return 1 - l_occlusion;
}

[numthreads(GTAO_THREAD_GROUP_SIZE, GTAO_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    // Output
    RWTexture2D<float> l_ao_texture = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float2 l_uv_position = p_dispatch_thread_id.xy + 0.5f;
    float2 l_uv = l_uv_position / (float2)g_push_constants.m_dst_resolution;

    float l_depth = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).r;

    float l_z_vs = get_view_depth_from_depth(l_depth, l_camera.m_z_near, l_camera.m_z_far);

    // Fade calculation
    float l_fade_ratio = 1.0f;
    float l_far_threshold = g_push_constants.m_far_fade_out_threshold;
    float l_fade_range = max(g_push_constants.m_far_fade_out_range, 1.0f);
    float l_delta = max(l_far_threshold - l_z_vs, 0.0f);
    l_fade_ratio *= min(l_delta / l_fade_range, 1.0f);

    // Fade near, avoid wrong ao in ads mode
    const float l_min_ads_dist = g_push_constants.m_near_fade_in_begin;
    const float l_near_ao_fade_dist = g_push_constants.m_near_fade_in_end;
    l_fade_ratio *= saturate((l_z_vs - l_min_ads_dist) / (l_near_ao_fade_dist - l_min_ads_dist));

    if (l_fade_ratio - FADE_DELTA < 0)
    {
        l_ao_texture[p_dispatch_thread_id.xy] = 1.0f;

        return;
    }

    float2 l_tan_half_fov_xy = get_tangent_half_fov_from_projection_matrix(l_camera.m_proj);
    float3 l_position_vs = get_view_position(l_uv, l_z_vs, l_tan_half_fov_xy);

    float3 l_normal_vs = get_view_normal(l_uv);

    const uint2 l_xy_id = p_dispatch_thread_id.xy & INTERLEAVE_POSITION_VS_MASK;
    const float2 l_pixel_percent = bindless_tex2d_load(g_push_constants.m_jitter_texture_srv, uint3(l_xy_id, 0)).xy;

    float l_ao = compute_coarse_ao(l_uv_position, l_pixel_percent, l_position_vs, l_normal_vs, g_push_constants.m_inv_dst_resolution,
                                   l_camera.m_z_near, l_camera.m_z_far, l_tan_half_fov_xy, g_push_constants.m_sample_aspect_ratio,
                                   g_push_constants.m_near_radius, g_push_constants.m_far_radius, g_push_constants.m_near_horizon_falloff,
                                   g_push_constants.m_far_horizon_falloff, g_push_constants.m_near_thickness, g_push_constants.m_far_thickness,
                                   g_push_constants.m_fade_start, g_push_constants.m_fade_speed, g_push_constants.m_half_project_scale);

    float l_result = saturate(1.0f - g_push_constants.m_intensity * l_ao);
    l_result = lerp(1.0f, l_result, l_fade_ratio);

    l_ao_texture[p_dispatch_thread_id.xy] = l_result;
}
