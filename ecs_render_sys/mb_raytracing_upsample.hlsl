// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// TODO: these are temporary defines for developing upsampling
//#define MB_RAYTRACING_LINEAR_UPSAMPLING
//#define MB_RAYTRACING_DIFFUSE_GI_ONLY

// CBV
ConstantBuffer<cb_push_raytracing_upsampling_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 nearest_depth_upsample(in float2 p_uv, in uint2 p_dst_pixel_coords)
{
    Texture2D<float4> l_accumulation_buffer = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_srv];
    Texture2D<float> l_depth_buffer = ResourceDescriptorHeap[g_push_constants.m_depth_texture_srv];

    // Fetch 2x2 quad of low-res pixels
    float4 l_accumulated_radiance_r = l_accumulation_buffer.GatherRed   ((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv, int2(0, 0));
    float4 l_accumulated_radiance_g = l_accumulation_buffer.GatherGreen ((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv, int2(0, 0));
    float4 l_accumulated_radiance_b = l_accumulation_buffer.GatherBlue  ((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv, int2(0, 0));
    float4 l_accumulated_radiance_a = l_accumulation_buffer.GatherAlpha ((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv, int2(0, 0));

    float4 l_accumulated_radiance_0 = float4(l_accumulated_radiance_r.x, l_accumulated_radiance_g.x, l_accumulated_radiance_b.x, l_accumulated_radiance_a.x);
    float4 l_accumulated_radiance_1 = float4(l_accumulated_radiance_r.y, l_accumulated_radiance_g.y, l_accumulated_radiance_b.y, l_accumulated_radiance_a.y);
    float4 l_accumulated_radiance_2 = float4(l_accumulated_radiance_r.z, l_accumulated_radiance_g.z, l_accumulated_radiance_b.z, l_accumulated_radiance_a.z);
    float4 l_accumulated_radiance_3 = float4(l_accumulated_radiance_r.w, l_accumulated_radiance_g.w, l_accumulated_radiance_b.w, l_accumulated_radiance_a.w);

    // Fetch high-res reference depth
    float l_reference_depth = l_depth_buffer.Load(uint3(p_dst_pixel_coords, 0));

    // Compute depth difference between reference values and all other samples
    float4 l_depth_diff = float4(l_reference_depth - l_accumulated_radiance_0.w,
                                 l_reference_depth - l_accumulated_radiance_1.w,
                                 l_reference_depth - l_accumulated_radiance_2.w,
                                 l_reference_depth - l_accumulated_radiance_3.w);
    l_depth_diff = abs(l_depth_diff);

    // If there is no big discontinuity - use linear sampling
    if (all(l_depth_diff < 0.0001f))
    {
        return l_accumulation_buffer.SampleLevel((SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], p_uv, 0).xyz;
    }

    // Pick the sample with the closest depth
    float4 l_result = l_accumulated_radiance_0;
    if (abs(l_reference_depth - l_result.w) > l_depth_diff.y)
    {
        l_result = l_accumulated_radiance_1;
    }
    if (abs(l_reference_depth - l_result.w) > l_depth_diff.z)
    {
        l_result = l_accumulated_radiance_2;
    }
    if (abs(l_reference_depth - l_result.w) > l_depth_diff.w)
    {
        l_result = l_accumulated_radiance_3;
    }

    return l_result.xyz;
}

[numthreads(RAYTRACING_UPSAMPLING_THREAD_GROUP_SIZE, RAYTRACING_UPSAMPLING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the texture
    if (p_dispatch_thread_id.x >= g_push_constants.m_dst_resolution_x ||
        p_dispatch_thread_id.y >= g_push_constants.m_dst_resolution_y)
    {
        return;
    }

    // UAVs
    RWTexture2D<float4> l_output                = ResourceDescriptorHeap[g_push_constants.m_output_uav];

    // SRVs
    Texture2D<float4> l_direct_lighting_rt      = ResourceDescriptorHeap[g_push_constants.m_direct_lighting_srv];
    Texture2D<float4> l_diffuse_reflectance_rt  = ResourceDescriptorHeap[g_push_constants.m_diffuse_reflectance_srv];
    Texture2D<float4> l_accumulation_buffer     = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_srv];

    // Ray-tracing can be computed in fractional coordinates(1, 1/2, 1/4, ...)
    uint2 l_full_res_coords = p_dispatch_thread_id.xy;
    uint2 l_full_res_dim = uint2(g_push_constants.m_dst_resolution_x, g_push_constants.m_dst_resolution_y);

    // Make sure UVs are in pixel centers
    float2 l_full_res_uv = (l_full_res_coords + 0.5f) / l_full_res_dim;

#if defined(MB_RAYTRACING_LINEAR_UPSAMPLING)
    float3 l_accumulated_radiance = l_accumulation_buffer.SampleLevel((SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_full_res_uv, 0).xyz;
#else
    float3 l_accumulated_radiance = nearest_depth_upsample(l_full_res_uv, l_full_res_coords.xy);
#endif

    // SSAO
    float l_ssao = 1.0f;
    if (g_push_constants.m_ssao_rt_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_ssao = bindless_tex2d_sample_level(g_push_constants.m_ssao_rt_srv, (SamplerState) SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_full_res_uv).r;
    }

#if defined(MB_RAYTRACING_DIFFUSE_GI)
    float3 l_direct_lighting = l_direct_lighting_rt.Load(int3(l_full_res_coords.xy, 0)).xyz;
    l_direct_lighting = unpack_lighting(l_direct_lighting);

    // When DiffuseGI is enabled - we also getting diffuse reflectance
    float3 l_diffuse_reflectance = l_diffuse_reflectance_rt.Load(int3(l_full_res_coords.xy, 0)).xyz;

    // Combine GI with direct lighting
#if !defined(MB_RAYTRACING_DIFFUSE_GI_ONLY)
    float3 l_lighting = l_direct_lighting + l_ssao * l_diffuse_reflectance * l_accumulated_radiance;
    //float3 l_lighting = l_ssao * l_accumulated_radiance;
#else
    float3 l_lighting = l_accumulated_radiance;
#endif

    l_output[l_full_res_coords.xy] = float4(pack_lighting(l_lighting), 1.0f);
#else
        float l_inv_acc_frame = 1.0f / (float) (g_push_constants.m_acc_frame_index + 1.0f);
        l_output[l_full_res_coords.xy] = float4(pack_lighting(l_accumulated_radiance) * l_inv_acc_frame, 1.0f);
#endif
    }
