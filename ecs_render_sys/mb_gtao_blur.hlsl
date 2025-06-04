// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

#define KERNEL_RADIUS 4

#define USE_DEPTH_SLOPE 1
#define USE_ADAPTIVE_SAMPLING 0

// Push constants
ConstantBuffer<cb_push_gtao_blur_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

// Group shared memory
groupshared float2 g_ao_depth_cache[GTAO_THREAD_GROUP_SIZE * GTAO_THREAD_GROUP_SIZE + KERNEL_RADIUS * KERNEL_RADIUS];

float2 sample_ao_depth(float2 p_uv, uint p_depth_texture_srv, uint p_ao_texture_srv, float p_near, float p_far)
{
    float l_depth = bindless_tex2d_sample_level(p_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_uv).r;

#if HALF_RESOLUTION
    float l_ao = bindless_tex2d_sample_level(p_ao_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv).r;
#else
    float l_ao = bindless_tex2d_sample_level(p_ao_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], p_uv).r;
#endif

    return float2(l_ao, get_view_depth_from_depth(l_depth, p_near, p_far) * 0.01);
}

float2 load_ao_depth_from_cache(int p_index)
{
    float2 l_ao_depth = g_ao_depth_cache[p_index];
    return l_ao_depth;
}

float2 load_ao_depth_from_cache_linear(int p_index_0, int p_index_1, float p_ratio)
{
    float2 l_ao_depth = (g_ao_depth_cache[p_index_0] + g_ao_depth_cache[p_index_1]) * p_ratio;
    return l_ao_depth;
}

struct center_pixel_data_t
{
    int m_index;
    float m_depth;
    float m_sharpness;
};

float cross_bilateral_weight(float p_r, float p_sample_depth, float p_depth_slope, center_pixel_data_t p_center)
{
    const float l_blur_sigma = ((float)KERNEL_RADIUS + 1.0) * 0.5;
    const float l_blur_falloff = 1.0 / (2.0 * l_blur_sigma * l_blur_sigma);

#if USE_DEPTH_SLOPE
    p_sample_depth -= p_depth_slope * p_r;
#endif

    float l_delta_z = (p_sample_depth - p_center.m_depth) * p_center.m_sharpness;

    return exp2(-p_r * p_r * l_blur_falloff - l_delta_z * l_delta_z);
}

void process_sample(float2 p_ao_z,
                    float p_r,
                    float p_depth_slope,
                    center_pixel_data_t p_center,
                    inout float p_total_ao,
                    inout float p_total_w)
{
    float l_ao = p_ao_z.x;
    float l_z = p_ao_z.y;

    float l_w = cross_bilateral_weight(p_r, l_z, p_depth_slope, p_center);
    p_total_ao += l_w * l_ao;
    p_total_w += l_w;
}

void process_radius(int p_r0,
                    int p_direction,
                    float p_depth_slope,
                    center_pixel_data_t p_center,
                    inout float p_total_ao,
                    inout float p_total_w)
{
#if USE_ADAPTIVE_SAMPLING
    float l_r = p_r0;

    [unroll]
    for (; l_r <= KERNEL_RADIUS / 2; l_r += 1)
    {
        int l_index = l_r * p_direction + p_center.m_index;
        float2 l_ao_z = load_ao_depth_from_cache(l_index);
        process_sample(l_ao_z, l_r, p_depth_slope, p_center, p_total_ao, p_total_w);
    }

    [unroll(KERNEL_RADIUS / 2 / 2)]
    for (; l_r <= KERNEL_RADIUS; l_r += 2)
    {
        int l_index = l_r * p_direction + p_center.m_index;
        float2 l_ao_z = load_ao_depth_from_cache_linear(l_index, l_index + p_direction, 0.5);
        process_sample(l_ao_z, l_r, p_depth_slope, p_center, p_total_ao, p_total_w);
    }
#else
    [unroll]
    for (float l_r = p_r0; l_r <= KERNEL_RADIUS; l_r += 1)
    {
        int l_index = l_r * p_direction + p_center.m_index;
        float2 l_ao_z = load_ao_depth_from_cache(l_index);
        process_sample(l_ao_z, l_r, p_depth_slope, p_center, p_total_ao, p_total_w);
    }
#endif
}

#if USE_DEPTH_SLOPE
void process_radius_with_depth_slope(int p_direction,
                                     center_pixel_data_t p_center,
                                     inout float p_total_ao,
                                     inout float p_total_w)
{
    float2 l_ao_depth = load_ao_depth_from_cache(p_center.m_index + p_direction);
    if (l_ao_depth.y <= 0.0f)
    {
        return;
    }

    float l_depth_slope = l_ao_depth.y - p_center.m_depth;

    process_sample(l_ao_depth, 1, l_depth_slope, p_center, p_total_ao, p_total_w);
    process_radius(2, p_direction, l_depth_slope, p_center, p_total_ao, p_total_w);
}
#endif

float compute_blur(int p_index, float2 p_ao_z, float p_blur_sharpness)
{
    center_pixel_data_t l_center;
    l_center.m_index = p_index;
    l_center.m_depth = p_ao_z.y;
    l_center.m_sharpness = p_blur_sharpness;

    float l_total_ao = p_ao_z.x;
    float l_total_w = 1.0f;

#if USE_DEPTH_SLOPE
    process_radius_with_depth_slope(1, l_center, l_total_ao, l_total_w);
    process_radius_with_depth_slope(-1, l_center, l_total_ao, l_total_w);
#else
    float l_depth_slope = 0;
    process_radius(1, 1, l_depth_slope, l_center, l_total_ao, l_total_w);
    process_radius(1, -1, -l_depth_slope, l_center, l_total_ao, l_total_w);
#endif

    return l_total_ao / l_total_w;
}

#if BLUR_X
[numthreads(GTAO_THREAD_GROUP_SIZE * GTAO_THREAD_GROUP_SIZE, 1, 1)]
void cs_main(int3 p_group_thread_id : SV_GroupThreadID, uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Output
    RWTexture2D<float> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    if (p_group_thread_id.x < KERNEL_RADIUS)
    {
        // Clamp out of bound samples that occur at image borders
        float2 l_uv = (uint2(p_dispatch_thread_id.x - KERNEL_RADIUS, p_dispatch_thread_id.y) + 0.5) * g_push_constants.m_inv_dst_resolution;
        g_ao_depth_cache[p_group_thread_id.x] = sample_ao_depth(l_uv,
                                                                g_push_constants.m_depth_texture_srv,
                                                                g_push_constants.m_ao_texture_srv,
                                                                l_camera.m_z_near,
                                                                l_camera.m_z_far);
    }

    if (p_group_thread_id.x >= GTAO_THREAD_GROUP_SIZE * GTAO_THREAD_GROUP_SIZE - KERNEL_RADIUS)
    {
        // Clamp out of bound samples that occur at image borders
        float2 l_uv = (uint2(p_dispatch_thread_id.x + KERNEL_RADIUS, p_dispatch_thread_id.y) + 0.5) * g_push_constants.m_inv_dst_resolution;
        g_ao_depth_cache[p_group_thread_id.x + 2 * KERNEL_RADIUS] = sample_ao_depth(l_uv,
                                                                                    g_push_constants.m_depth_texture_srv,
                                                                                    g_push_constants.m_ao_texture_srv,
                                                                                    l_camera.m_z_near,
                                                                                    l_camera.m_z_far);
    }

    // Clamp out of bound samples that occur at image borders
    g_ao_depth_cache[p_group_thread_id.x + KERNEL_RADIUS] = sample_ao_depth((p_dispatch_thread_id.xy + 0.5) * g_push_constants.m_inv_dst_resolution,
                                                                            g_push_constants.m_depth_texture_srv,
                                                                            g_push_constants.m_ao_texture_srv,
                                                                            l_camera.m_z_near,
                                                                            l_camera.m_z_far);

    GroupMemoryBarrierWithGroupSync();

    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    int l_index = p_group_thread_id.x + KERNEL_RADIUS;
    float2 l_ao_z = g_ao_depth_cache[l_index];
    if (l_ao_z.y >= g_push_constants.m_far_fade_out_threshold * 0.01)
    {
        l_rt[p_dispatch_thread_id.xy] = 1.0f;
        return;
    }

    float l_ao = compute_blur(l_index, l_ao_z, g_push_constants.m_blur_sharpness);
    l_rt[p_dispatch_thread_id.xy] = l_ao;
}
#elif BLUR_Y
[numthreads(1, GTAO_THREAD_GROUP_SIZE * GTAO_THREAD_GROUP_SIZE, 1)]
void cs_main(int3 p_group_thread_id : SV_GroupThreadID, uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Output
    RWTexture2D<float> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    if (p_group_thread_id.y < KERNEL_RADIUS)
    {
        // Clamp out of bound samples that occur at image borders
        float2 l_uv = (uint2(p_dispatch_thread_id.x, p_dispatch_thread_id.y - KERNEL_RADIUS) + 0.5) * g_push_constants.m_inv_dst_resolution;
        g_ao_depth_cache[p_group_thread_id.y] = sample_ao_depth(l_uv,
                                                                g_push_constants.m_depth_texture_srv,
                                                                g_push_constants.m_ao_texture_srv,
                                                                l_camera.m_z_near,
                                                                l_camera.m_z_far);
    }

    if (p_group_thread_id.y >= GTAO_THREAD_GROUP_SIZE * GTAO_THREAD_GROUP_SIZE - KERNEL_RADIUS)
    {
        // Clamp out of bound samples that occur at image borders
        float2 l_uv = (uint2(p_dispatch_thread_id.x, p_dispatch_thread_id.y + KERNEL_RADIUS) + 0.5) * g_push_constants.m_inv_dst_resolution;
        g_ao_depth_cache[p_group_thread_id.y + 2 * KERNEL_RADIUS] = sample_ao_depth(l_uv,
                                                                                    g_push_constants.m_depth_texture_srv,
                                                                                    g_push_constants.m_ao_texture_srv,
                                                                                    l_camera.m_z_near,
                                                                                    l_camera.m_z_far);
    }

    // Clamp out of bound samples that occur at image borders
    g_ao_depth_cache[p_group_thread_id.y + KERNEL_RADIUS] = sample_ao_depth((p_dispatch_thread_id.xy + 0.5) * g_push_constants.m_inv_dst_resolution,
                                                                            g_push_constants.m_depth_texture_srv,
                                                                            g_push_constants.m_ao_texture_srv,
                                                                            l_camera.m_z_near,
                                                                            l_camera.m_z_far);

    GroupMemoryBarrierWithGroupSync();

    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    int m_index = p_group_thread_id.y + KERNEL_RADIUS;
    float2 l_ao_z = g_ao_depth_cache[m_index];
    if (l_ao_z.y >= g_push_constants.m_far_fade_out_threshold * 0.01)
    {
        l_rt[p_dispatch_thread_id.xy] = 1.0f;
        return;
    }

    float l_ao = compute_blur(m_index, l_ao_z, g_push_constants.m_blur_sharpness);
    l_rt[p_dispatch_thread_id.xy] = pow(clamp(l_ao, 1e-5, 1), g_push_constants.m_power_exponent);
}
#endif
