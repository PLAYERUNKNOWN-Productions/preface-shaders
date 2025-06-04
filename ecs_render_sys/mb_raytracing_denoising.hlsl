// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"

// Different denosing modes
#define MT_DENOISING_POISSON        (0)
#define MT_DENOISING_A_TROUS        (1)
#define MT_DENOISING_A_TROUS_BOX    (2)
#define MT_DENOISING_MODE MT_DENOISING_A_TROUS_BOX

// Per frame poisson kernel rotation
#define MT_ROTATE_POISSON_KERNEL

// Generated with https://github.com/bartwronski/PoissonSamplingGenerator
static const uint s_sample_num = 33;
static const float2 s_poisson_samples[s_sample_num] =
{
    float2(-0.639177717692996f, 0.6817836159377788f),
    float2(-0.43558405304261616f, -0.8190425157471635f),
    float2(0.8344933853801978f, 0.1001015732577139f),
    float2(-0.08104036049002612f, 0.016739289336124077f),
    float2(-0.912711939665125f, -0.039970070166963516f),
    float2(0.17135343759624036f, -0.45947139094990547f),
    float2(0.13733871235206196f, 0.9753708190198431f),
    float2(0.7044428399965664f, 0.6084704294169025f),
    float2(0.6618476561508143f, -0.6130068262199246f),
    float2(-0.49030526893710524f, -0.3264270081304258f),
    float2(0.3923545452094815f, 0.27169019349152734f),
    float2(-0.8250759563777875f, -0.4144512506782365f),
    float2(0.23468941243654032f, 0.6713499004010706f),
    float2(-0.33256544043837727f, 0.36805529183450103f),
    float2(-0.7006947206638415f, 0.3025561842607366f),
    float2(-0.3575203056166003f, 0.8701157949257953f),
    float2(0.0447398935705463f, -0.9791359528836265f),
    float2(0.4057776326442341f, -0.8115023137263795f),
    float2(0.9922736469047126f, -0.07033443386273058f),
    float2(-0.9205704223788547f, 0.2754276970361924f),
    float2(-0.3982143075247596f, 0.0472071402269093f),
    float2(0.41563606563589156f, 0.027341018107576585f),
    float2(-0.03683678250013172f, -0.7088265507407878f),
    float2(0.041355786809185474f, 0.5407020586314644f),
    float2(-0.24656967857897535f, -0.3026334019853586f),
    float2(-0.5157099715429905f, -0.5812092689390425f),
    float2(0.8585180942591087f, -0.4039716406411972f),
    float2(0.4612839661045865f, 0.6080820996170297f),
    float2(0.4459501287739186f, -0.5451610245727035f),
    float2(0.17300049374680293f, -0.11326683839894873f),
    float2(0.46867696461867814f, -0.19031475270904016f),
    float2(0.12580570351493137f, 0.19197533075550458f),
    float2(0.0f, 0.0f),
};

// CBV
ConstantBuffer<cb_push_raytracing_denoising_t> g_push_constants  : register(REGISTER_PUSH_CONSTANTS);

float gaussian_weight(float p_x, float p_y, float p_sigma)
{
    return exp(-0.5f * (p_x * p_x + p_y * p_y) / (p_sigma * p_sigma)) / (2.0f * M_PI * p_sigma * p_sigma);
}

void accumulate_sample( Texture2D<float4> p_accumulation_buffer_src,
                        Texture2D<float3> p_normals_texture,
                        ConstantBuffer<cb_camera_t> p_camera,
                        uint3 p_sample_coords,
                        float3 p_reference_normal,
                        float p_reference_depth,
                        float p_reference_z,
                        float p_custom_sample_weight,
                        float p_linear_depth_threshold,
                        inout float3 p_avg_radiance,
                        inout float p_total_weight)
{
    // Load outside of texture will return 0 and resulting in (l_depth_weight == 0) -> sample is rejected
    float4 l_radiance_depth = p_accumulation_buffer_src.Load(p_sample_coords);
    float3 l_normal = p_normals_texture.Load(p_sample_coords);

    // Proj vs Linear Z
#if 0
    float l_depth_weight = abs(l_radiance_depth.w - p_reference_depth) < 0.0001f;
#else
    float l_z = get_view_depth_from_depth(l_radiance_depth.w, p_camera.m_z_near, p_camera.m_z_far);
    float l_depth_weight = abs(l_z - p_reference_z) < p_linear_depth_threshold;
#endif

    // Normals
    float l_normal_weight = 1.0f;
#if 1
    l_normal_weight = dot(normalize(p_reference_normal), normalize(l_normal)) > 0.9f ? 1.0f : 0.01f;
#endif

    // Combined sample weight
    float l_sample_weight = l_depth_weight * l_normal_weight * p_custom_sample_weight;

    p_avg_radiance += l_radiance_depth.xyz * l_sample_weight;
    p_total_weight += l_sample_weight;
}

[numthreads(RAYTRACING_DENOISING_THREAD_GROUP_SIZE, RAYTRACING_DENOISING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint2 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (p_dispatch_thread_id.x >= g_push_constants.m_dst_resolution_x ||
        p_dispatch_thread_id.y >= g_push_constants.m_dst_resolution_y)
    {
        return;
    }

    RWTexture2D<float4> l_accumulation_buffer_dst = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_uav];

    Texture2D<float4> l_accumulation_buffer_src = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_srv];
    Texture2D<float> l_frame_count_texture_read = ResourceDescriptorHeap[g_push_constants.m_raytracing_frame_count_texture_srv];
    Texture2D<float3> l_normals_texture         = ResourceDescriptorHeap[g_push_constants.m_raytracing_normals_srv];

    float l_reference_depth         = l_accumulation_buffer_src[p_dispatch_thread_id.xy].w;
    float l_num_accumulated_frames  = 255.0f * l_frame_count_texture_read[p_dispatch_thread_id.xy];
    float3 l_reference_normal       = l_normals_texture[p_dispatch_thread_id.xy];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float l_reference_z = get_view_depth_from_depth(l_reference_depth, l_camera.m_z_near, l_camera.m_z_far);

    float l_z_threshold = lerp(0.02f, 10.0f, saturate(l_reference_z / 100.0f));

    // Based on papers:
    // 1) https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9985-exploring-ray-traced-future-in-metro-exodus.pdf
    // 2) https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf
#if (MT_DENOISING_MODE == MT_DENOISING_POISSON)
    const int c_kernel_size = 4;

    float3 l_avg_radiance = 0;
    float l_total_weight = 0;

    float l_kernel_scale = 16.0f / (l_num_accumulated_frames + 1.0f);
    l_kernel_scale = max(l_kernel_scale, 4.0f);

    for (int l_index = 0; l_index < s_sample_num; ++l_index)
    {
#if defined(MT_ROTATE_POISSON_KERNEL)
        float2 l_sample_offset = float2(s_poisson_samples[l_index].x * g_push_constants.m_poisson_kernel_rot_cos - s_poisson_samples[l_index].y * g_push_constants.m_poisson_kernel_rot_sin,
                                        s_poisson_samples[l_index].x * g_push_constants.m_poisson_kernel_rot_sin + s_poisson_samples[l_index].y * g_push_constants.m_poisson_kernel_rot_cos);
#else
        float2 l_sample_offset = s_poisson_samples[l_index];
#endif

        uint3 l_sample_coords = uint3(p_dispatch_thread_id.xy + l_kernel_scale * l_sample_offset + 0.5f, 0);

        accumulate_sample(l_accumulation_buffer_src,
                          l_normals_texture,
                          l_camera,
                          l_sample_coords,
                          l_reference_normal,
                          l_reference_depth,
                          l_reference_z,
                          1.0f,
                          l_z_threshold,
                          l_avg_radiance,
                          l_total_weight);
    }
#endif

#if (MT_DENOISING_MODE == MT_DENOISING_A_TROUS_BOX)
    // Variable kernel scale based on the number of accumulated frames
    int c_kernel_size = lerp(6, 2, l_num_accumulated_frames / 32.0f);

    float3 l_avg_radiance = 0;
    float l_total_weight = 0;

    // TODO: give kernel a progressive step to cover bigger area
    // This is not energy conserving, but give better visual results
    uint l_sample_step_scale = g_push_constants.m_denoising_iteration == 0 ? 1 : 3 * (g_push_constants.m_denoising_iteration);

    for (int l_x = -c_kernel_size; l_x <= c_kernel_size; ++l_x)
    {
        for (int l_y = -c_kernel_size; l_y <= c_kernel_size; ++l_y)
        {
            uint3 l_sample_coords = uint3(p_dispatch_thread_id.xy + int2(l_x, l_y) * l_sample_step_scale, 0);

            accumulate_sample(l_accumulation_buffer_src,
                              l_normals_texture,
                              l_camera,
                              l_sample_coords,
                              l_reference_normal,
                              l_reference_depth,
                              l_reference_z,
                              1.0f,
                              l_z_threshold,
                              l_avg_radiance,
                              l_total_weight);
        }
    }
#endif

    // Based on paper: https://jo.dreggn.org/home/2010_atrous.pdf
#if (MT_DENOISING_MODE == MT_DENOISING_A_TROUS)
    // A-Trous wavelet coeficients
    const int c_kernel_size = 5;
    const float c_a_trous_weights[c_kernel_size] = { 1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f };

    float3 l_avg_radiance = 0;
    float l_total_weight = 0;

    // According to paper each next iteration is 2 ^ (i - 1) wider
    uint l_sample_step_scale = g_push_constants.m_denoising_iteration == 0 ? 1 : 2u << (g_push_constants.m_denoising_iteration - 1);

    for (int l_x = -2; l_x <= 2; ++l_x)
    {
        for (int l_y = -2; l_y <= 2; ++l_y)
        {
            uint3 l_sample_coords = uint3(p_dispatch_thread_id.xy + int2(l_x, l_y) * l_sample_step_scale, 0);

            float l_a_trous_weight = c_a_trous_weights[l_x + 2] * c_a_trous_weights[l_y + 2];

            accumulate_sample(l_accumulation_buffer_src,
                              l_normals_texture,
                              l_camera,
                              l_sample_coords,
                              l_reference_normal,
                              l_reference_depth,
                              l_reference_z,
                              l_a_trous_weight,
                              l_z_threshold,
                              l_avg_radiance,
                              l_total_weight);
        }
    }
#endif

#if 1
    l_accumulation_buffer_dst[p_dispatch_thread_id.xy] = float4(l_avg_radiance / l_total_weight, l_reference_depth);
    //l_accumulation_buffer_dst[p_dispatch_thread_id.xy] = l_accumulation_buffer_src[p_dispatch_thread_id.xy];
#else
    l_accumulation_buffer_dst[p_dispatch_thread_id.xy] = float4(l_num_accumulated_frames.xxx, l_reference_depth);
    //l_accumulation_buffer_dst[p_dispatch_thread_id.xy] = float4(10000.0  *l_normals_texture[p_dispatch_thread_id.xy], l_reference_depth);
#endif
}
