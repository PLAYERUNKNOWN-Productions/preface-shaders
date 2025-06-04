// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_atmospheric_scattering_utils.hlsl"

ConstantBuffer<cb_push_generate_scattering_lut_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(GENERATE_SCATTERING_LUT_THREAD_GROUP_SIZE, GENERATE_SCATTERING_LUT_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get uv
    // uv.x: [0, 1], [altitude on the ground, altitude at the top of the atmosphere]
    // uv.y: [0, 1] to [-1, 1], [cos(PI), cos(0)], zenith angle
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5) / (float2)g_push_constants.m_dst_resolution;

    // UV to altitude and cosine theta
    float l_altitude = l_uv.x * g_push_constants.m_atmosphere_height;
    float l_cos_theta = -1.0f + l_uv.y * 2.0f;

    // Get density ratio of current point
    float2 l_density_ratio = exp(-l_altitude.xx / g_push_constants.m_density_scale_height);

    // Get optical depth along the light direction
    float3 l_ray_start = float3(0.0f, l_altitude + g_push_constants.m_planet_radius, 0.0f);
    float3 l_ray_dir = normalize(float3(sqrt(1.0f - l_cos_theta * l_cos_theta), l_cos_theta, 0.0f));
    float2 l_optical_depth = get_optical_depth_along_light_direction(l_ray_start, l_ray_dir, float3(0.0f, 0.0f, 0.0f),
                                                                     g_push_constants.m_planet_radius, g_push_constants.m_atmosphere_height,
                                                                     g_push_constants.m_density_scale_height, g_push_constants.m_sample_count);

    l_rt[p_dispatch_thread_id.xy] = float4(l_density_ratio, l_optical_depth * TO_FLOAT16_SCALE);
}
