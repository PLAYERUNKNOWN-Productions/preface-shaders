// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_LIGHTING_HLSL
#define MB_SHADER_LIGHTING_HLSL

#include "../shared_shaders/mb_shared_common.hlsl"

// PCF shader variants
// PCF_BOX_FILTER
// PCF_POISSON
// PCF_OPTIMIZED

#define MIN_PBR_ROUGHNESS 0.001f

// Specifies minimal reflectance for dielectrics (when metalness is zero)
#define MIN_DIELECTRICS_F0 0.04f

uint reverse_bits_32(uint p_bits)
{
    p_bits = (p_bits << 16) | (p_bits >> 16);
    p_bits = ((p_bits & 0x55555555) << 1) | ((p_bits & 0xAAAAAAAA) >> 1);
    p_bits = ((p_bits & 0x33333333) << 2) | ((p_bits & 0xCCCCCCCC) >> 2);
    p_bits = ((p_bits & 0x0F0F0F0F) << 4) | ((p_bits & 0xF0F0F0F0) >> 4);
    p_bits = ((p_bits & 0x00FF00FF) << 8) | ((p_bits & 0xFF00FF00) >> 8);

    return p_bits;
}

// Hammersley without random number
float2 hammersley(uint p_index, uint p_num_samples)
{
    return float2(float(p_index) / float(p_num_samples), reverse_bits_32(p_index) * 2.3283064365386963e-10);
}

// Hammersley with random number
float2 hammersley(uint p_index, uint p_num_samples, uint2 p_random)
{
    float l_eta1 = frac((float)p_index / p_num_samples + float(p_random.x & 0xffff) / (1 << 16));
    float l_eta2 = float(reverse_bits_32(p_index) ^ p_random.y) * 2.3283064365386963e-10; // float * 2^-32
    return float2(l_eta1, l_eta2);
}

// Potential improvements:
// This paper claim the Frisvad method produces inaccuracies when tangent is close to (0, 0, -1).
// Try usingn their method instead
// [Duff et al. 2017, "Building an Orthonormal Basis, Revisited"]
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
float3x3 build_tbn(float3 p_tangent_z)
{
    float3 l_up_vector = abs(p_tangent_z.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 l_tangent_x = normalize(cross(l_up_vector, p_tangent_z));
    float3 l_tangent_y = cross(p_tangent_z, l_tangent_x);
    return float3x3(l_tangent_x, l_tangent_y, p_tangent_z);
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 18
float3 importance_sample_cos(float2 p_u, float3x3 p_tangent_basis)
{
    // Cosine sampling
    float l_cos_theta = sqrt(max(0.0f, 1.0f - p_u.x));
    float l_sin_theta = sqrt(p_u.x);
    float l_phi = 2.0f * M_PI * p_u.y;

    // Transform from spherical into cartesian
    float3 l_l = float3(l_sin_theta * cos(l_phi), l_sin_theta * sin(l_phi), l_cos_theta);

    // Local to world
    l_l = mul(l_l, p_tangent_basis);

    return l_l;
}

// [Karis 2013, "Real Shading in Unreal Engine 4"]
float3 importance_sample_ggx(float2 p_u, float p_alpha, float3x3 p_tangent_basis)
{
    // GGX NDF sampling
    float l_phi = 2.0f * M_PI * p_u.x;
    float l_cos_theta = sqrt((1.0f - p_u.y) / (1.0f + (p_alpha * p_alpha - 1.0f) * p_u.y));
    float l_sin_theta = sqrt(max(0.0f, 1.0f - l_cos_theta * l_cos_theta));

    // Transform from spherical into cartesian
    float3 l_h = float3(l_sin_theta * cos(l_phi), l_sin_theta * sin(l_phi), l_cos_theta);

    // Local to world
    l_h = mul(l_h, p_tangent_basis);

    return l_h;
}

float3 base_color_to_diffuse_reflectance(float3 p_base_color, float p_metalness)
{
    return p_base_color * (1.0f - p_metalness);
}

float3 base_color_to_specular_f0(float3 p_base_color, float p_metalness)
{
    return lerp(MIN_DIELECTRICS_F0, p_base_color, p_metalness);
}

// Trowbridge-Reitz GGX
// [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
// [Karis 2013, "Real Shading in Unreal Engine 4"]
float distribution_ggx(float p_noh, float p_a)
{
    float l_a2 = p_a * p_a;
    float l_noh2 = p_noh * p_noh;
    float l_denominator = l_noh2 * (l_a2 - 1.0f) + 1.0f;
    l_denominator = l_denominator * l_denominator; // No need to add 1e-7f here since we already clamped the roughness

    return l_a2 * M_INV_PI / l_denominator;
}

// Fresnel-Schlick approximation
// [An Inexpensive BDRF Model for Physically based Rendering]
float3 fresnel_reflectance_schlick(float3 p_f0, float p_f90, float p_u)
{
    return p_f0 + (p_f90 - p_f0) * pow(1.0f - p_u, 5.0f);
}

// Fresnel-Schlick approximation
// [An Inexpensive BDRF Model for Physically based Rendering]
float3 fresnel_reflectance_schlick(float3 p_f0, float p_voh)
{
    return fresnel_reflectance_schlick(p_f0, 1.0f, p_voh);
}

// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
// Frostbite & Filament's GGX Smith Joint implementation
// TODO: check [Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"]
float vis_smith_joint(float p_nov, float p_nol, float p_a)
{
    float l_a2 = p_a * p_a;
    float l_vis_smith_v = p_nol * sqrt(p_nov * (p_nov - p_nov * l_a2) + l_a2);
    float l_vis_smith_l = p_nov * sqrt(p_nol * (p_nol - p_nol * l_a2) + l_a2);
    return 0.5f / (l_vis_smith_v + l_vis_smith_l + 1e-5f);
}

// Disney Diffuse (no 1/PI)
// [Burley 2012, "Physically-Based Shading at Disney"]
float disney_diffuse_no_pi(float p_nov, float p_nol, float p_loh, float p_linear_roughness)
{
    float l_fd90 = 0.5f + 2.0f * p_loh * p_loh * p_linear_roughness;
    const float3 l_f0 = float3(1.0f, 1.0f, 1.0f);

    // Two Schlick Fresnel term
    float l_light_scatter = fresnel_reflectance_schlick(l_f0, l_fd90, p_nol).r;
    float l_view_scatter = fresnel_reflectance_schlick(l_f0, l_fd90, p_nov).r;

    return l_light_scatter * l_view_scatter;
}

// Renormalized Disney Diffuse (no 1/PI)
// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 1
float renormalized_disney_diffuse_no_pi(float p_nov, float p_nol, float p_loh, float p_linear_roughness)
{
    float l_energy_bias = lerp(0.0f, 0.5f, p_linear_roughness);
    float l_energy_factor = lerp(1.0f, 1.0f / 1.51f, p_linear_roughness);

    float l_fd90 = l_energy_bias + 2.0f * p_loh * p_loh * p_linear_roughness;
    const float3 l_f0 = float3(1.0f, 1.0f, 1.0f);

    // Two Schlick Fresnel term
    float l_light_scatter = fresnel_reflectance_schlick(l_f0, l_fd90, p_nol).r;
    float l_view_scatter = fresnel_reflectance_schlick(l_f0, l_fd90, p_nov).r;

    return l_light_scatter * l_view_scatter * l_energy_factor;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 23
float3 get_diffuse_dominant_dir(float3 p_n, float3 p_v, float p_alpha)
{
    float l_nov = saturate(dot(p_n, p_v));
    float l_a = 1.02341f * p_alpha - 1.51174f;
    float l_b = -0.511705f * p_alpha + 0.755868f;
    float l_lerp_factor = saturate((l_nov * l_a + l_b) * p_alpha);

    return normalize(lerp(p_n, p_v, l_lerp_factor));
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 22
float3 get_specular_dominant_dir(float3 p_n, float3 p_r, float p_alpha)
{
    float l_smoothness = saturate(1 - p_alpha);
    float l_lerp_factor = l_smoothness * (sqrt(l_smoothness) + p_alpha);

    // The result is not normalized as we fetch in a cubemap
    return lerp(p_n, p_r, l_lerp_factor);
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 26
float compute_specular_occlusion(float p_nov, float p_ao, float p_alpha)
{
    return saturate(pow(p_nov + p_ao, exp2(-16.0f * p_alpha - 1.0f)) - 1.0f + p_ao);
}

// Clamp roughness and remap linear roughness to alpha
float clamp_and_remap_roughness(float p_linear_roughness)
{
    // Clamp roughness to avoid PBR precision problems
    p_linear_roughness = max(p_linear_roughness, MIN_PBR_ROUGHNESS);

    // Remap roughness
    float l_alpha = p_linear_roughness * p_linear_roughness;

    return l_alpha;
}

float3 ggx_direct_lighting(float3 p_normal, float3 p_view_dir, float3 p_light_dir,
                           float3 p_diffuse_reflectance, float3 p_specular_f0, float p_roughness)
{
    // Clamp and remap roughness
    float l_alpha = clamp_and_remap_roughness(p_roughness);

    // Prepare vectors
    float3 l_half = normalize(p_light_dir + p_view_dir);
    float l_nol = saturate(dot(p_normal, p_light_dir));
    float l_nov = saturate(dot(p_normal, p_view_dir));
    float l_noh = saturate(dot(p_normal, l_half));
    float l_voh = saturate(dot(p_view_dir, l_half));

    // Normal distribution function
    float l_d = distribution_ggx(l_noh, l_alpha);

    // Visibility function
    float l_v = vis_smith_joint(l_nov, l_nol, l_alpha);

    // Fresnel equation
    float3 l_f = fresnel_reflectance_schlick(p_specular_f0, l_voh);

    // Cook-Torrance BRDF
    float3 l_specular_lighting = l_d * l_v * l_f;

    // Lambert diffuse
    float3 l_diffuse_lighting = p_diffuse_reflectance * M_INV_PI;

    return (l_diffuse_lighting + l_specular_lighting) * l_nol;
}

struct omni_light_t
{
    float3 m_color;
    float3 m_position_ws_local; // m_param_0.xyz
    float  m_range;             // m_param_0.w
};

struct spot_light_t
{
    float3 m_color;
    float3 m_position_ws_local; // m_param_0.xyz
    float  m_range;             // m_param_0.w
    float3 m_direction;         // m_param_1.xyz
    float  m_angle_scale;       // m_param_1.w       angle_scale = 1.0f / max(1e-5f, (cos_inner - cos_outer))
    float  m_angle_offset;      // m_param_2.x       angle_offset = -cos_outer * angle_scale;
};

struct directional_light_t
{
    float3 m_color;
    float3 m_direction;         // m_param_0.xyz
};

omni_light_t get_omni_light_param(cb_light_t p_light)
{
    omni_light_t l_omni_light;
    l_omni_light.m_color                = p_light.m_color;
    l_omni_light.m_position_ws_local    = p_light.m_param_0.xyz;
    l_omni_light.m_range                = p_light.m_param_0.w;
    return l_omni_light;
}

spot_light_t get_spot_light_param(cb_light_t p_light)
{
    spot_light_t l_spot_light;
    l_spot_light.m_color                = p_light.m_color;
    l_spot_light.m_position_ws_local    = p_light.m_param_0.xyz;
    l_spot_light.m_range                = p_light.m_param_0.w;
    l_spot_light.m_direction            = p_light.m_param_1.xyz;
    l_spot_light.m_angle_scale          = p_light.m_param_1.w;
    l_spot_light.m_angle_offset         = p_light.m_param_2.x;
    return l_spot_light;
}

directional_light_t get_directional_light_param(cb_light_t p_light)
{
    directional_light_t l_directional_light;
    l_directional_light.m_color     = p_light.m_color;
    l_directional_light.m_direction = p_light.m_param_0.xyz;
    return l_directional_light;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 4
float smooth_distance_attenuation(float p_squared_distance, float p_inv_sqr_atten_radius)
{
    float l_factor = p_squared_distance * p_inv_sqr_atten_radius;
    float l_smooth_factor = saturate(1.0f - l_factor * l_factor);
    return l_smooth_factor * l_smooth_factor;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 4 + UE hack
float get_distance_attenuation(float3 p_unnormalized_light_dir, float p_inv_sqr_atten_radius)
{
    float l_sqr_distance = dot(p_unnormalized_light_dir, p_unnormalized_light_dir);
    float l_attenuation = 1.0f / (l_sqr_distance + 1); // +1 to avoid inf
    l_attenuation *= smooth_distance_attenuation(l_sqr_distance, p_inv_sqr_atten_radius);
    return l_attenuation;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 4
float get_angle_attenuation(float3 p_to_light, float3 p_light_dir, float p_light_angle_scale, float p_light_angle_offset)
{
    // On the CPU
    // float l_light_angle_scale = 1.0f / max(1e-5, (l_cos_inner - l_cos_outer));
    // float l_light_angle_offset = -l_cos_outer * l_light_angle_scale;

    float l_cd = dot(p_light_dir, p_to_light);
    float l_attenuation = saturate(l_cd * p_light_angle_scale + p_light_angle_offset);

    // Smooth the transition
    l_attenuation *= l_attenuation;

    return l_attenuation;
}

float omni_light_atten(float3 p_position_world, float3 p_light_pos, float p_light_range)
{
    // Distance attenuation
    float3 l_light_dir = p_light_pos - p_position_world;
    float l_inv_sqr_atten_radius = 1.0f / (p_light_range * p_light_range);
    float l_atten = get_distance_attenuation(l_light_dir, l_inv_sqr_atten_radius);

    return l_atten;
}

float spot_light_atten(float3 p_position_world, float3 p_light_pos, float p_light_range, float3 p_spot_dir,
                       float p_light_angle_scale, float p_light_angle_offset)
{
    // Distance attenuation
    float3 l_light_dir = p_light_pos - p_position_world;
    float l_inv_sqr_atten_radius = 1.0f / (p_light_range * p_light_range);
    float l_atten = get_distance_attenuation(l_light_dir, l_inv_sqr_atten_radius);

    l_light_dir = normalize(l_light_dir);

    // Angle attenuation
    float3 l_spot_dir = normalize(p_spot_dir);
    l_atten *= get_angle_attenuation(l_light_dir, l_spot_dir, p_light_angle_scale, p_light_angle_offset);

    return l_atten;
}

float linear_roughness_to_mipmap_level(float p_linear_roughness, float p_mip_count)
{
    return p_linear_roughness * p_mip_count;
}

float mipmap_level_to_linear_roughness(float p_mipmap_level, float p_mip_count)
{
    return p_mipmap_level / p_mip_count;
}

// From Unity
uint get_ibl_runtime_filter_sample_count(uint p_mip_level)
{
    uint l_sample_count = 0;

    switch (p_mip_level)
    {
        case 1: l_sample_count = 21; break;
        case 2: l_sample_count = 34; break;
        case 3: l_sample_count = 55; break;
        case 4: l_sample_count = 89; break;
        case 5: l_sample_count = 89; break;
        case 6: l_sample_count = 89; break;
    }

    return l_sample_count;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 18
float4 integrate_dfg(float3 p_n, float3 p_v, float p_linear_roughness, uint p_sample_count)
{
    // Clamp roughness to avoid PBR precision problems
    p_linear_roughness = max(p_linear_roughness, MIN_PBR_ROUGHNESS);

    // Remap roughness
    float l_alpha = p_linear_roughness * p_linear_roughness;

    float l_nov = saturate(dot(p_n, p_v));
    float4 l_acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Get tangent basis
    float3x3 l_tangent_basis = build_tbn(p_n);

    for (uint l_i = 0; l_i < p_sample_count; l_i++)
    {
        float2 l_u = hammersley(l_i, p_sample_count);

        float3 l_h = importance_sample_ggx(l_u, l_alpha, l_tangent_basis);
        float3 l_l = 2.0f * dot(p_v, l_h) * l_h - p_v; // Convert sample from half angle to incident angle
        float l_nol = saturate(dot(p_n, l_l));

        // Specular GGX DFG pre-integration
        if (l_nol > 0.0f)
        {
            float l_noh = saturate(dot(p_n, l_h));
            float l_voh = saturate(dot(p_v, l_h));
            float l_vis = vis_smith_joint(l_nov, l_nol, l_alpha);

            // l_weight_over_pdf return the weight (without the p_diffuse_albedo term) over pdf
            float l_weight_over_pdf = 4.0f * l_vis * l_nol * l_voh / l_noh;

            // Recombine at runtime with: (l_f0 * l_weight_over_pdf * (1 - l_fc) + l_f90 * l_weight_over_pdf * l_fc ) with l_fc = (1 - l_voh) ^ 5
            float l_fc = pow(1.0f - l_voh, 5.0f);
            l_acc.x += (1.0f - l_fc) * l_weight_over_pdf;
            l_acc.y += l_fc * l_weight_over_pdf;
        }

        // For Disney we still use a cosine importance sampling, true Disney importance sampling imply a look up table
        l_l = importance_sample_cos(l_u, l_tangent_basis);
        l_nol = saturate(dot(p_n, l_l));

        // Diffuse Disney pre-integration
        if (l_nol > 0.0f)
        {
            float3 l_h = normalize(l_l + p_v);
            float l_loh = dot(l_l, l_h);
            float l_disney_diffuse = disney_diffuse_no_pi(l_nov, l_nol, l_loh, p_linear_roughness);

            // Importance sampling weight for each sample
            // l_pdf = l_nol / M_PI
            // l_weight = l_fr * l_nol with l_fr = p_diffuse_albedo / M_PI
            // weight over pdf is:
            // l_weight_over_pdf = (p_diffuse_albedo / M_PI) * l_nol / (l_nol / M_PI)
            // l_weight_over_pdf = p_diffuse_albedo
            // p_diffuse_albedo is apply outside the function
            float l_weight_over_pdf = 1.0f;

            l_acc.z += l_disney_diffuse * l_weight_over_pdf;
        }
    }

    return l_acc / p_sample_count;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 19
float4 integrate_specular_ld(uint p_cubemap_index, float3 p_n, float3 p_v, float p_alpha, float p_inv_omega_p, uint p_sample_count)
{
    float3 l_acc = float3(0.0f, 0.0f, 0.0f);
    float l_acc_weight = 0;

    // Get tangent basis
    float3x3 l_tangent_basis = build_tbn(p_n);

    for (uint l_i = 0; l_i < p_sample_count; l_i++)
    {
        float2 l_u = hammersley(l_i, p_sample_count);

        float3 l_h = importance_sample_ggx(l_u, p_alpha, l_tangent_basis);
        float3 l_l = 2.0f * dot(p_v, l_h) * l_h - p_v;

        float l_nol = saturate(dot(p_n, l_l));
        if (l_nol > 0.0f)
        {
            float l_mip_level = 0;

            // Prefiltered BRDF importance sampling
            {
                float l_noh = saturate(dot(p_n, l_h));
                float l_loh = saturate(dot(l_l, l_h));

                // Use pre-filtered importance sampling (i.e use lower mipmap
                // level for fetching sample with low probability in order
                // to reduce the variance).
                // (Reference : GPU Gem3: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html)
                //
                // Since we pre - integrate the result for normal direction ,
                // p_n == p_v and then l_noh == l_loh. This is why the BRDF pdf
                // can be simplifed from :
                // l_pdf = l_d * l_noh / (4 * l_loh) to l_pdf = l_d / 4;
                //
                // - l_omega_s : Solid angle associated to a sample
                // - l_omega_p : Solid angle associated to a pixel of the cubemap
                float l_pdf = distribution_ggx(l_noh, p_alpha) * l_noh / (4 * l_loh);
                float l_omega_s = 1.0f / (p_sample_count * l_pdf); // Solid angle associated to a sample
                // p_inv_omega_p is precomputed on CPU and provide as a parameter of the function
                // float l_omega_p = 4.0f * M_PI / (6.0f * l_cubemap_width * l_cubemap_width);

                // Clamp is not necessary as the hardware will do it
                // mipLevel = clamp(0.5f * log2(l_omega_s * p_inv_omega_p), 0, p_mipmap_count);
                l_mip_level = 0.5f * log2(l_omega_s * p_inv_omega_p);
            }

            float3 l_val = bindless_texcube_sample_level(p_cubemap_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_l, l_mip_level).rgb;

            // See p64 equation (53) of moving Frostbite to PBR v3 for the extra l_nol here (both in weight and value)
            l_acc += l_val * l_nol;
            l_acc_weight += l_nol;
        }
    }

    return float4(l_acc * (1.0f / l_acc_weight), 1.0f);
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 20
float4 integrate_diffuse_ld(uint p_cubemap_index, float3 p_n, uint p_sample_count, uint p_mip_level)
{
    float3 l_acc = float3(0.0f, 0.0f, 0.0f);

    // Get tangent basis
    float3x3 l_tangent_basis = build_tbn(p_n);

    for (uint l_i = 0; l_i < p_sample_count; l_i++)
    {
        float2 l_u = hammersley(l_i, p_sample_count);
        float3 l_l = importance_sample_cos(l_u, l_tangent_basis);

        float l_nol = saturate(dot(p_n, l_l));
        if (l_nol > 0)
        {
            l_acc += bindless_texcube_sample_level(p_cubemap_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_l, p_mip_level).rgb;
        }
    }

    return float4(l_acc / p_sample_count, 1.0f);
}

// Helper function for optimized PCF
float sample_shadow_map(float2 p_base_uv, float p_u, float p_v, float2 p_inv_shadow_map_size, float p_depth, uint p_shadow_texture_srv)
{
    float2 l_uv = p_base_uv + float2(p_u, p_v) * p_inv_shadow_map_size;
    Texture2D l_shadow_texture = ResourceDescriptorHeap[p_shadow_texture_srv];
    return l_shadow_texture.SampleCmpLevelZero((SamplerComparisonState)SamplerDescriptorHeap[SAMPLER_COMPARISON_LINEAR_GREATER], l_uv, p_depth);
}

// optimized PCF
// Theory: http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/
// Original implementation: https://github.com/TheRealMJP/Shadows/blob/master/Shadows/Mesh.hlsl, under MIT license
float pcf_optimized(float2 p_base_uv, float p_depth, float p_depth_bias, float2 p_shadow_map_size, float2 p_inv_shadow_map_size, uint p_shadow_texture_srv)
{
    float l_light_depth = p_depth + p_depth_bias; // Assuming reversed z

    float2 l_uv = p_base_uv * p_shadow_map_size;

    float2 l_base_uv;
    l_base_uv.x = floor(l_uv.x + 0.5f);
    l_base_uv.y = floor(l_uv.y + 0.5f);

    float l_s = (l_uv.x + 0.5f - l_base_uv.x);
    float l_t = (l_uv.y + 0.5f - l_base_uv.y);

    l_base_uv -= float2(0.5f, 0.5f);
    l_base_uv *= p_inv_shadow_map_size;

    // Filter size == 5
    float l_uw0 = (4 - 3 * l_s);
    float l_uw1 = 7;
    float l_uw2 = (1 + 3 * l_s);

    float l_u0 = (3 - 2 * l_s) / l_uw0 - 2;
    float l_u1 = (3 + l_s) / l_uw1;
    float l_u2 = l_s / l_uw2 + 2;

    float l_vw0 = (4 - 3 * l_t);
    float l_vw1 = 7;
    float l_vw2 = (1 + 3 * l_t);

    float l_v0 = (3 - 2 * l_t) / l_vw0 - 2;
    float l_v1 = (3 + l_t) / l_vw1;
    float l_v2 = l_t / l_vw2 + 2;

    float l_sum = 0;

    l_sum += l_uw0 * l_vw0 * sample_shadow_map(l_base_uv, l_u0, l_v0, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw1 * l_vw0 * sample_shadow_map(l_base_uv, l_u1, l_v0, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw2 * l_vw0 * sample_shadow_map(l_base_uv, l_u2, l_v0, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);

    l_sum += l_uw0 * l_vw1 * sample_shadow_map(l_base_uv, l_u0, l_v1, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw1 * l_vw1 * sample_shadow_map(l_base_uv, l_u1, l_v1, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw2 * l_vw1 * sample_shadow_map(l_base_uv, l_u2, l_v1, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);

    l_sum += l_uw0 * l_vw2 * sample_shadow_map(l_base_uv, l_u0, l_v2, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw1 * l_vw2 * sample_shadow_map(l_base_uv, l_u1, l_v2, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);
    l_sum += l_uw2 * l_vw2 * sample_shadow_map(l_base_uv, l_u2, l_v2, p_inv_shadow_map_size, l_light_depth, p_shadow_texture_srv);

    return l_sum * 1.0f / 144;
}

// Helper function for optimized PCF GSM version
float sample_shadow_map_gsm(float2 p_base_uv, float p_u, float p_v, float2 p_inv_shadow_map_size, Texture2D<float> p_shadow_texture)
{
    float2 l_uv = p_base_uv + float2(p_u, p_v) * p_inv_shadow_map_size;
    return p_shadow_texture.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv, 0);
}

float pcf_optimized_gsm(float2 p_base_uv, float2 p_shadow_map_size, float2 p_inv_shadow_map_size, Texture2D<float> p_shadow_texture)
{
    float2 l_uv = p_base_uv * p_shadow_map_size;

    float2 l_base_uv;
    l_base_uv.x = floor(l_uv.x + 0.5f);
    l_base_uv.y = floor(l_uv.y + 0.5f);

    float l_s = (l_uv.x + 0.5f - l_base_uv.x);
    float l_t = (l_uv.y + 0.5f - l_base_uv.y);

    l_base_uv -= float2(0.5f, 0.5f);
    l_base_uv *= p_inv_shadow_map_size;

    // Filter size == 5
    float l_uw0 = (4 - 3 * l_s);
    float l_uw1 = 7;
    float l_uw2 = (1 + 3 * l_s);

    float l_u0 = (3 - 2 * l_s) / l_uw0 - 2;
    float l_u1 = (3 + l_s) / l_uw1;
    float l_u2 = l_s / l_uw2 + 2;

    float l_vw0 = (4 - 3 * l_t);
    float l_vw1 = 7;
    float l_vw2 = (1 + 3 * l_t);

    float l_v0 = (3 - 2 * l_t) / l_vw0 - 2;
    float l_v1 = (3 + l_t) / l_vw1;
    float l_v2 = l_t / l_vw2 + 2;

    float l_sum = 0;

    l_sum += l_uw0 * l_vw0 * sample_shadow_map_gsm(l_base_uv, l_u0, l_v0, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw1 * l_vw0 * sample_shadow_map_gsm(l_base_uv, l_u1, l_v0, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw2 * l_vw0 * sample_shadow_map_gsm(l_base_uv, l_u2, l_v0, p_inv_shadow_map_size, p_shadow_texture);

    l_sum += l_uw0 * l_vw1 * sample_shadow_map_gsm(l_base_uv, l_u0, l_v1, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw1 * l_vw1 * sample_shadow_map_gsm(l_base_uv, l_u1, l_v1, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw2 * l_vw1 * sample_shadow_map_gsm(l_base_uv, l_u2, l_v1, p_inv_shadow_map_size, p_shadow_texture);

    l_sum += l_uw0 * l_vw2 * sample_shadow_map_gsm(l_base_uv, l_u0, l_v2, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw1 * l_vw2 * sample_shadow_map_gsm(l_base_uv, l_u1, l_v2, p_inv_shadow_map_size, p_shadow_texture);
    l_sum += l_uw2 * l_vw2 * sample_shadow_map_gsm(l_base_uv, l_u2, l_v2, p_inv_shadow_map_size, p_shadow_texture);

    return l_sum * 1.0f / 144;
}

#ifndef PCF_NUM
#define PCF_NUM 9
#endif

#if PCF_NUM == 4

static const float2 g_pcf_offset[PCF_NUM] =
{
    float2(-1.0f, -1.0f),
    float2( 1.0f, -1.0f),
    float2( 1.0f,  1.0f),
    float2(-1.0f,  1.0f),
};

#elif PCF_NUM == 9

static const float2 g_pcf_offset[PCF_NUM] =
{
    float2(-1.0f, -1.0f),
    float2(-1.0f,  0.0f),
    float2(-1.0f,  1.0f),
    float2( 0.0f, -1.0f),
    float2( 0.0f,  0.0f),
    float2( 0.0f,  1.0f),
    float2( 1.0f, -1.0f),
    float2( 1.0f,  0.0f),
    float2( 1.0f,  1.0f),
};

#endif

float pcf_box_filter(float2 p_base_uv, float p_depth, float p_depth_bias, float2 p_inv_shadow_map_size, uint p_shadow_texture_srv)
{
    float l_light_depth = p_depth + p_depth_bias; // Assuming reversed z

    float l_sum = 0;

    for (int l_i = 0; l_i < PCF_NUM; l_i++)
    {
        float2 l_uv = p_base_uv + g_pcf_offset[l_i] * p_inv_shadow_map_size;
        Texture2D l_shadow_texture = ResourceDescriptorHeap[p_shadow_texture_srv];
        l_sum += l_shadow_texture.SampleCmpLevelZero((SamplerComparisonState)SamplerDescriptorHeap[SAMPLER_COMPARISON_LINEAR_GREATER], l_uv, l_light_depth);
    }

    return l_sum /= PCF_NUM;
}

float pcf_box_filter_gsm(float2 p_base_uv, float2 p_inv_shadow_map_size, Texture2D<float> p_shadow_texture)
{
    float l_sum = 0;

    for (int l_i = 0; l_i < PCF_NUM; l_i++)
    {
        float2 l_uv = p_base_uv + g_pcf_offset[l_i] * p_inv_shadow_map_size;
        l_sum += p_shadow_texture.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv, 0);
    }

    return l_sum /= PCF_NUM;
}

#define PCF_NUM_POISSON 16
static const float2 g_pcf_offset_possion[PCF_NUM_POISSON] =
{
    float2(-0.94201624,  -0.39906216),
    float2( 0.94558609,  -0.76890725),
    float2(-0.094184101, -0.92938870),
    float2( 0.34495938,   0.29387760),
    float2(-0.91588581,   0.45771432),
    float2(-0.81544232,  -0.87912464),
    float2(-0.38277543,   0.27676845),
    float2( 0.97484398,   0.75648379),
    float2( 0.44323325,  -0.97511554),
    float2( 0.53742981,  -0.47373420),
    float2(-0.26496911,  -0.41893023),
    float2( 0.79197514,   0.19090188),
    float2(-0.24188840,   0.99706507),
    float2(-0.81409955,   0.91437590),
    float2( 0.19984126,   0.78641367),
    float2( 0.14383161,  -0.14100790),
};

// PCF with Poisson Disk Sampling
float pcf_poisson(float2 p_base_uv, float p_depth, float p_depth_bias, float p_filter_radius, float2 p_inv_shadow_map_size, uint p_shadow_texture_srv)
{
    float l_light_depth = p_depth + p_depth_bias; // Assuming reversed z

    float l_sum = 0;

    for (int l_i = 0; l_i < PCF_NUM_POISSON; l_i++)
    {
        float2 l_uv = p_base_uv + g_pcf_offset_possion[l_i] * p_inv_shadow_map_size * p_filter_radius;
        Texture2D l_shadow_texture = ResourceDescriptorHeap[p_shadow_texture_srv];
        l_sum += l_shadow_texture.SampleCmpLevelZero((SamplerComparisonState)SamplerDescriptorHeap[SAMPLER_COMPARISON_LINEAR_GREATER], l_uv, l_light_depth);
    }

    return l_sum /= PCF_NUM_POISSON;
}

// PCF with Poisson Disk Sampling, adjusted version for gsm
float pcf_poisson_gsm(float2 p_base_uv, float p_filter_radius, float2 p_inv_shadow_map_size, Texture2D<float> p_shadow_texture)
{
    float l_sum = 0;

    for (int l_i = 0; l_i < PCF_NUM_POISSON; l_i++)
    {
        float2 l_uv = p_base_uv + g_pcf_offset_possion[l_i] * p_inv_shadow_map_size * p_filter_radius;
        l_sum += p_shadow_texture.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_uv, 0);
    }

    return l_sum /= PCF_NUM_POISSON;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 24
float3 evaluate_ibl_diffuse(float3 p_n, float3 p_v, float p_linear_roughness, uint p_dfg_texture_index, uint p_ld_texture_index)
{
    // Clamp roughness to avoid PBR precision problems
    p_linear_roughness = max(p_linear_roughness, MIN_PBR_ROUGHNESS);

    // Remap roughness
    float l_alpha = p_linear_roughness * p_linear_roughness;

    float3 l_dominant_n = get_diffuse_dominant_dir(p_n, p_v, l_alpha);
    float3 l_diffuse_lighting = bindless_texcube_sample_level(p_ld_texture_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_dominant_n).rgb;

    float l_nov = saturate(dot(p_n, p_v));
    float l_diff_f = bindless_tex2d_sample_level(p_dfg_texture_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float2(l_nov, p_linear_roughness), 0).z;

    return l_diffuse_lighting * l_diff_f;
}

// [Lagarde et al. 2014, "Moving Frostbite to PBR"] : Listing 24
float3 evaluate_ibl_specular(float3 p_n, float3 p_v, float p_linear_roughness, uint p_dfg_texture_index, uint p_dfg_texture_size,
                             uint p_ld_texture_index, float p_ld_mip_count, float3 p_f0, float p_f90)
{
    // Clamp roughness to avoid PBR precision problems
    p_linear_roughness = max(p_linear_roughness, MIN_PBR_ROUGHNESS);

    // Remap roughness
    float l_alpha = p_linear_roughness * p_linear_roughness;

    float3 l_r = reflect(-p_v, p_n);
    float3 l_dominant_r = get_specular_dominant_dir(p_n, l_r, l_alpha);
    float l_mip_level = linear_roughness_to_mipmap_level(p_linear_roughness, p_ld_mip_count);
    float3 l_pre_ld = bindless_texcube_sample_level(p_ld_texture_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_dominant_r, l_mip_level).rgb;

    // Sample pre-integrated DFG
    // Fc = (1 - H * L) ^ 5
    // PreIntegratedDFG.r = Gv * (1 - Fc)
    // PreIntegratedDFG.g = Gv * Fc
    float l_nov = saturate(dot(p_n, p_v));
    l_nov = max(l_nov, 0.5f / p_dfg_texture_size);
    float2 l_pre_dfg = bindless_tex2d_sample_level(p_dfg_texture_index, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], float2(l_nov, p_linear_roughness), 0).xy;

    // Rebuild the function
    // L * D * (f0 * Gv * (1 - Fc) + Gv * Fc) * cosTheta / (4 * NoL * NoV)
    return l_pre_ld * (p_f0 * l_pre_dfg.x + p_f90 * l_pre_dfg.y);
}

float fake_earth_direct_shadow_brdf(float3 p_planet_normal, float3 p_light_dir)
{
#if ENABLE_FAKE_EARTH_SHADOW_TERM
    float l_coef = smoothstep(-0.01f, 0.0, dot(p_planet_normal, p_light_dir));
    return l_coef * l_coef;
#else
    return 1.0f;
#endif
}

float fake_earth_ibl_shadow_brdf(float3 p_planet_normal, cb_light_list_t p_light_list)
{
#if ENABLE_FAKE_EARTH_SHADOW_TERM_IBL
    float3 l_first_direct_light_dir = float3(1.0, 0, 0);
    for (int l_i = 0; l_i < MB_MAX_LIGHTS; l_i++)
    {
        cb_light_t l_light = p_light_list.m_light_list[l_i];
        if (l_light.m_type == LIGHT_TYPE_DIRECTIONAL) // Directional light
        {
            directional_light_t l_directional_light = get_directional_light_param(l_light);
            l_first_direct_light_dir = l_directional_light.m_direction;
            break;
        }
    }

    return smoothstep(-0.08, 0.3, dot(p_planet_normal, l_first_direct_light_dir));
#else
    return 1.0f;
#endif
}

// http://en.wikipedia.org/wiki/Film_speed
float luminance_to_ev100(float p_luminance)
{
    const float l_k = 12.5; // Reflected-light meter calibration constant
    return log2(p_luminance * 100.0 / l_k);
}

// http://en.wikipedia.org/wiki/Film_speed
float ev100_to_luminance(float p_ev100)
{
    // Compute the maximum luminance possible with H_sbs sensitivity
    // lum = 78 / ( S * q ) * N ^ 2 / t
    //     = 78 / ( S * q ) * 2 ^ EV100
    //     = 78 / (100 * 0.65) * 2 ^ EV100
    //     = 1.2 * 2 ^ EV100
    float l_luminance = 1.2 * pow(2.0, p_ev100);
    return l_luminance;
}

float ev100_to_exposure(float p_ev100)
{
    float l_max_luminance = ev100_to_luminance(p_ev100);
    return 1.0 / l_max_luminance;
}

float compute_luminance_adaptation(float p_previous_luminance, float p_current_luminance, float p_speed_dark_to_light, float p_speed_light_to_dark, float p_delta_time)
{
    float l_delta = p_current_luminance - p_previous_luminance;

    // The adaptation to dark takes longer time
    float l_speed = l_delta > 0.0 ? p_speed_dark_to_light : p_speed_light_to_dark;

    // The process of adaptation, an exponential decay function
    float l_factor = 1.0f - exp2(-p_delta_time * l_speed);
    float l_adapted_luminance = p_previous_luminance + l_delta * l_factor;

    return l_adapted_luminance;
}

// Calculate gsm shadows
float gsm_shadows(uint p_gsm_srv,
                  float3 p_position_ws_local,
                  float4x4 p_view_local_proj)
{
    float l_shadow = 1.0f;

#ifdef MB_GSM_ENABLED
    float3 l_uv_depth = pos_to_uv_depth(p_position_ws_local, p_view_local_proj);

    if(all(l_uv_depth.xy >= 0) && all(l_uv_depth.xy <= 1.0f))// && p_gsm_srv != RAL_NULL_BINDLESS_INDEX
    {
        Texture2D<float> l_gsm = ResourceDescriptorHeap[p_gsm_srv];
        float2 l_gsm_dimensions;
        l_gsm.GetDimensions(l_gsm_dimensions.x, l_gsm_dimensions.y);

#   if defined(PCF_BOX_FILTER)
        l_shadow = pcf_box_filter_gsm(l_uv_depth.xy, rcp(l_gsm_dimensions), l_gsm);
#   elif defined(PCF_POISSON)
        l_shadow = pcf_poisson_gsm(l_uv_depth.xy, 1.0f, rcp(l_gsm_dimensions), l_gsm);
#   elif defined(PCF_OPTIMIZED)
        l_shadow = pcf_optimized_gsm(l_uv_depth.xy, l_gsm_dimensions, rcp(l_gsm_dimensions), l_gsm);
#   else // No PCF
        l_shadow = l_gsm.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv_depth.xy, 0);
#   endif
    }
#endif

    return l_shadow;
}

// Calculate direct shadows
float direct_shadows(uint p_shadow_caster_count,
                     uint p_shadow_caster_srv,
                     float3 p_position_ws_local,
                     float3 p_triangle_normal_ws,
                     directional_light_t p_directional_light,
                     uint p_gsm_srv,
                     float4x4 p_gsm_camera_view_local_proj)
{
    float l_csm_shadow = 1.0f;
    float l_gsm_shadow = gsm_shadows(p_gsm_srv, p_position_ws_local, p_gsm_camera_view_local_proj);

    if(p_shadow_caster_srv != RAL_NULL_BINDLESS_INDEX)
    {
        StructuredBuffer<sb_shadow_caster_t> l_shadow_caster_buffer = ResourceDescriptorHeap[p_shadow_caster_srv];

        // Select cascade
        for(uint l_shadow_caster_index = 0; l_shadow_caster_index < p_shadow_caster_count; ++l_shadow_caster_index)
        {
            // Get shadow caster
            sb_shadow_caster_t l_shadow_caster = l_shadow_caster_buffer[l_shadow_caster_index];

            // Get shadow casting camera
            ConstantBuffer<cb_camera_t> l_shadow_camera = ResourceDescriptorHeap[l_shadow_caster.m_shadow_camera_cbv];

            // Get vertex normal
            float3 l_vertex_normal = normalize(p_triangle_normal_ws);

            // Find a normal offset
            float l_light_cos_angle = dot(l_vertex_normal, -p_directional_light.m_direction);
            float l_normal_offset = saturate(1.0f - l_light_cos_angle);
            float l_shadow_texel_size = sqrt(2.0f) * l_shadow_caster.m_shadow_texel_size;
            l_normal_offset *= l_shadow_texel_size; // Get texel size

            // Transform position into shadow camera's local space
            float3 l_shadow_camera_offset = float3( l_shadow_caster.m_shadow_camera_offset_x,
                                                    l_shadow_caster.m_shadow_camera_offset_y,
                                                    l_shadow_caster.m_shadow_camera_offset_z);
            float3 l_shadow_local_pos = p_position_ws_local - l_shadow_camera_offset;
            float3 l_shadow_local_pos_offset = l_shadow_local_pos + l_normal_offset * normalize(l_vertex_normal);

            // We have two options: apply offset in 3D or only in UV space
            // UV space is more expensive but should give better results
            // The current implementation uses 3D space
            float4 l_pos_vs = mul(float4(l_shadow_local_pos_offset, 1.0f), l_shadow_camera.m_view_local);
            float4 l_pos_cs = mul(l_pos_vs, l_shadow_camera.m_proj);
            float3 l_proj_pos = l_pos_cs.xyz / l_pos_cs.w;

            // Get SM UV coordinates
            float2 l_uv = l_proj_pos.xy * 0.5 + 0.5;
            l_uv.y = 1.0f - l_uv.y;

            // Early-exit if we are outside of the cascade frustum
            if(any(l_uv < 0) || any(l_uv > 1.0f) || (l_proj_pos.z < 0) || (l_proj_pos.z > 1.0))
            {
                continue;
            }

            // Introduce small constant depth bias to fight depth texture and view-proj matrix imprecision
            float l_depth_bias = l_shadow_caster.m_shadow_constant_depth_bias;

#if defined(DEBUG_VISUALIZE_CASCADES)
            return 2 + l_shadow_caster_index;
#endif

#if defined(PCF_BOX_FILTER)
            l_csm_shadow = pcf_box_filter(l_uv, l_proj_pos.z, l_depth_bias, l_shadow_caster.m_inv_shadow_map_size, l_shadow_caster.m_shadow_texture_srv);
#elif defined(PCF_POISSON)
            l_csm_shadow = pcf_poisson(l_uv, l_proj_pos.z, l_depth_bias, 1.0f, l_shadow_caster.m_inv_shadow_map_size, l_shadow_caster.m_shadow_texture_srv);
#elif defined(PCF_OPTIMIZED)
            l_csm_shadow = pcf_optimized(l_uv, l_proj_pos.z, l_depth_bias, l_shadow_caster.m_shadow_map_size, l_shadow_caster.m_inv_shadow_map_size, l_shadow_caster.m_shadow_texture_srv);
#else // No PCF
            // Manual point filtering
            Texture2D l_shadow_texture = ResourceDescriptorHeap[l_shadow_caster.m_shadow_texture_srv];
            float l_depth_test = l_shadow_texture.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv, 0).x;
            l_csm_shadow = l_depth_test < l_proj_pos.z + l_depth_bias;
#endif

            // Blendout last cascade. This will be the transition between CSM and GSM/NoShadows
            if(l_shadow_caster_index == p_shadow_caster_count - 1)
            {
                const float c_blendout_strength = 0.1f;
                float l_start_value = 1.0f - c_blendout_strength;
                float2 l_lerp = saturate((abs(l_uv * 2.0f - 1.0f) - l_start_value) / c_blendout_strength);
                float l_blend_val = max(l_lerp.x, l_lerp.y);
                l_csm_shadow = max(l_csm_shadow, l_blend_val);
            }

            break;
        }

        sb_shadow_caster_t l_last_shadow_caster = l_shadow_caster_buffer[p_shadow_caster_count - 1];

        // Fadeout CSM cascade based on distance. This prevents the cascade from casting shadows into huge distances.
        const float c_blendout_distance_scale = 3.0f;
        float l_start_range = l_last_shadow_caster.m_max_shadow_casting_distance;
        float l_end_range = l_start_range * c_blendout_distance_scale;
        float l_lerp_value = saturate((length(p_position_ws_local) - l_start_range) / (l_end_range - l_start_range));
        l_csm_shadow = max(l_csm_shadow, l_lerp_value);
    }

    return min(l_csm_shadow, l_gsm_shadow);
}

// Calculate direct lighting
float3 light_direct(cb_light_list_t p_light_list, float3 p_normal_ws, float3 p_view_dir, float p_roughness,
                    float3 p_diffuse_reflectance, float3 p_specular_f0, float3 p_position_ws_local, float3 p_planet_normal,
                    uint p_shadow_caster_count, uint p_shadow_caster_srv,
                    uint p_global_shadow_map_srv, float4x4 p_gsm_camera_view_local_proj)
{
    float3 l_direct_lighting = 0;

    for (int l_i = 0; l_i < MB_MAX_LIGHTS; l_i++)
    {
        cb_light_t l_light = p_light_list.m_light_list[l_i];
        if (l_light.m_type == LIGHT_TYPE_DIRECTIONAL) // Directional light
        {
#if ENABLE_DIRECTIONAL_LIGHT
            directional_light_t l_directional_light = get_directional_light_param(l_light);
            float3 l_lighting = ggx_direct_lighting(p_normal_ws, p_view_dir, l_directional_light.m_direction,
                                                    p_diffuse_reflectance, p_specular_f0, p_roughness);

            l_lighting *= l_directional_light.m_color * fake_earth_direct_shadow_brdf(p_planet_normal, l_directional_light.m_direction);

            float l_shadow = direct_shadows(p_shadow_caster_count,
                                            p_shadow_caster_srv,
                                            p_position_ws_local,
                                            p_normal_ws, // TODO: p_triangle_normal_ws,
                                            l_directional_light,
                                            p_global_shadow_map_srv,
                                            p_gsm_camera_view_local_proj);

#if defined(DEBUG_VISUALIZE_CASCADES)
            if (l_shadow == 2)
            {
                l_lighting *= float3(1, 0, 0);
            }
            else if (l_shadow == 3)
            {
                l_lighting *= float3(0, 1, 0);
            }
            else if (l_shadow == 4)
            {
                l_lighting *= float3(0, 0, 1);
            }
            else if (l_shadow == 5)
            {
                l_lighting *= float3(1, 1, 0);
            }
#else
            // Apply shadowing
            l_lighting *= l_shadow;
#endif

            l_direct_lighting += l_lighting;
#endif
        }
        else if (l_light.m_type == LIGHT_TYPE_OMNI) // Omni-directional light
        {
#if ENABLE_OMNI_LIGHT
            omni_light_t l_omni_light = get_omni_light_param(l_light);
            float l_atten = omni_light_atten(p_position_ws_local, l_omni_light.m_position_ws_local, l_omni_light.m_range);
            float3 l_light_dir = normalize(l_omni_light.m_position_ws_local - p_position_ws_local);
            float3 l_lighting = ggx_direct_lighting(p_normal_ws, p_view_dir, l_light_dir, p_diffuse_reflectance, p_specular_f0, p_roughness);

            l_direct_lighting += l_lighting * l_omni_light.m_color * l_atten;
#endif
        }
        else if (l_light.m_type == LIGHT_TYPE_SPOT) // Spot light
        {
#if ENABLE_SPOT_LIGHT
            spot_light_t l_spot_light = get_spot_light_param(l_light);
            float l_atten = spot_light_atten(p_position_ws_local, l_spot_light.m_position_ws_local, l_spot_light.m_range,
                                             l_spot_light.m_direction, l_spot_light.m_angle_scale, l_spot_light.m_angle_offset);
            float3 l_light_dir = normalize(l_spot_light.m_position_ws_local - p_position_ws_local);
            float3 l_lighting = ggx_direct_lighting(p_normal_ws, p_view_dir, l_light_dir, p_diffuse_reflectance, p_specular_f0, p_roughness);

            l_direct_lighting += l_lighting * l_spot_light.m_color * l_atten;
#endif
        }
        else // LIGHT_TYPE_NULL
        {
            break;
        }
    }

    return l_direct_lighting;
}

float3 ggx_direct_lighting_translucent(float3 p_normal, float3 p_view_dir, float3 p_light_dir,
    float3 p_diffuse_reflectance, float3 p_specular_f0, float p_roughness,
    float p_shadow)
{
    // The amount of light that continues through an object
    const float l_translucency_amount = 0.2;

    // Clamp and remap roughness
    float l_alpha = clamp_and_remap_roughness(p_roughness);

    // Prepare vectors
    float3 l_half = normalize(p_light_dir + p_view_dir);
    float l_nol = dot(p_normal, p_light_dir);
    float l_nol_translucent = (abs(l_nol) * l_translucency_amount + saturate(l_nol)) * (1.0 - l_translucency_amount);
    float l_nov = saturate(dot(p_normal, p_view_dir));
    float l_noh = saturate(dot(p_normal, l_half));
    float l_voh = saturate(dot(p_view_dir, l_half));

    // Normal distribution function
    float l_d = distribution_ggx(l_noh, l_alpha);

    // Visibility function
    float l_v = vis_smith_joint(l_nov, l_nol_translucent, l_alpha);

    // Fresnel equation
    float3 l_f = fresnel_reflectance_schlick(p_specular_f0, l_voh);

    // Cook-Torrance BRDF
    float3 l_specular_lighting = l_d * l_v * l_f * p_shadow;

    // Lambert diffuse
    float3 l_diffuse_lighting = p_diffuse_reflectance * M_INV_PI * (p_shadow * 0.975 + 0.025);

    return (l_diffuse_lighting + l_specular_lighting) * l_nol_translucent;
}

float3 light_direct_translucent(cb_light_list_t p_light_list, float3 p_normal_ws, float3 p_view_dir, float p_roughness,
    float3 p_diffuse_reflectance, float3 p_specular_f0, float3 p_position_ws_local, float3 p_planet_normal,
    uint p_shadow_caster_count, uint p_shadow_caster_srv,
    uint p_global_shadow_map_srv, float4x4 p_gsm_camera_view_local_proj)
{
    float3 l_direct_lighting = 0;

    for (int l_i = 0; l_i < MB_MAX_LIGHTS; l_i++)
    {
        cb_light_t l_light = p_light_list.m_light_list[l_i];
        if (l_light.m_type == LIGHT_TYPE_DIRECTIONAL) // Directional light
        {
#if ENABLE_DIRECTIONAL_LIGHT
            directional_light_t l_directional_light = get_directional_light_param(l_light);

            float l_shadow = direct_shadows(p_shadow_caster_count,
                                            p_shadow_caster_srv,
                                            p_position_ws_local,
                                            p_normal_ws, // TODO: p_triangle_normal_ws,
                                            l_directional_light,
                                            p_global_shadow_map_srv,
                                            p_gsm_camera_view_local_proj);

#   if defined(DEBUG_VISUALIZE_CASCADES)
            float3 l_shadow_color = 1;
            if (l_shadow == 2)
            {
                l_shadow_color = float3(1, 0, 0);
            }
            else if (l_shadow == 3)
            {
                l_shadow_color = float3(0, 1, 0);
            }
            else if (l_shadow == 4)
            {
                l_shadow_color = float3(0, 0, 1);
            }
            else if (l_shadow == 5)
            {
                l_shadow_color = float3(1, 1, 0);
            }
            l_shadow = 1.0f;
#   endif

            float3 l_lighting = ggx_direct_lighting_translucent(p_normal_ws, p_view_dir, l_directional_light.m_direction,
                p_diffuse_reflectance, p_specular_f0, p_roughness,
                l_shadow);

            l_lighting *= l_directional_light.m_color * fake_earth_direct_shadow_brdf(p_planet_normal, l_directional_light.m_direction);

#   if defined(DEBUG_VISUALIZE_CASCADES)
            // Apply shadowing
            l_lighting *= l_shadow_color;
#   endif

            l_direct_lighting += l_lighting;
#endif
        }
        else if (l_light.m_type == LIGHT_TYPE_OMNI) // Omni-directional light
        {
#if ENABLE_OMNI_LIGHT
            omni_light_t l_omni_light = get_omni_light_param(l_light);
            float l_atten = omni_light_atten(p_position_ws_local, l_omni_light.m_position_ws_local, l_omni_light.m_range);
            float3 l_light_dir = normalize(l_omni_light.m_position_ws_local - p_position_ws_local);
            float3 l_lighting = ggx_direct_lighting_translucent(p_normal_ws, p_view_dir, l_light_dir, p_diffuse_reflectance, p_specular_f0, p_roughness,
                1.0);

            l_direct_lighting += l_lighting * l_omni_light.m_color * l_atten;
#endif
        }
        else if (l_light.m_type == LIGHT_TYPE_SPOT) // Spot light
        {
#if ENABLE_SPOT_LIGHT
            spot_light_t l_spot_light = get_spot_light_param(l_light);
            float l_atten = spot_light_atten(p_position_ws_local, l_spot_light.m_position_ws_local, l_spot_light.m_range,
                l_spot_light.m_direction, l_spot_light.m_angle_scale, l_spot_light.m_angle_offset);
            float3 l_light_dir = normalize(l_spot_light.m_position_ws_local - p_position_ws_local);
            float3 l_lighting = ggx_direct_lighting_translucent(p_normal_ws, p_view_dir, l_light_dir, p_diffuse_reflectance, p_specular_f0, p_roughness,
                1.0);

            l_direct_lighting += l_lighting * l_spot_light.m_color * l_atten;
#endif
        }
        else // LIGHT_TYPE_NULL
        {
            break;
        }
    }

    return l_direct_lighting;
}

// TEMPORARY // DIRECT LIGHT WITH TRANSLUCENCY APPROXIMATION // END

// Calculate IBL
float3 light_ibl(float3 p_normal_ws, float3 p_view_dir, float p_roughness, float p_ao, float3 p_diffuse_reflectance,
                 float3 p_specular_f0, float p_exposure_value, uint p_dfg_texture_srv, uint p_diffuse_ld_texture_srv,
                 uint p_specular_ld_texture_srv, uint p_dfg_texture_size, uint p_specular_ld_mip_count,
                 cb_light_list_t p_light_list, float3 p_planet_normal, float3x3 p_align_ground_rotation)
{
    float l_ibl_intensity_multiplier = ev100_to_luminance(p_exposure_value);

    // IBL is baked with identity rotation. Depending where camera position is - probe needs to be rotated.
    // Instead of rotating the probe we rotate the vectors
    p_normal_ws = mul(p_normal_ws, p_align_ground_rotation);
    p_view_dir = mul(p_view_dir, p_align_ground_rotation);

    // IBL diffuse
    float3 l_ibl_diffuse = 0;
    if (p_dfg_texture_srv != RAL_NULL_BINDLESS_INDEX && p_diffuse_ld_texture_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_ibl_diffuse = evaluate_ibl_diffuse(
            p_normal_ws, p_view_dir, p_roughness, p_dfg_texture_srv,
            p_diffuse_ld_texture_srv) * p_diffuse_reflectance * l_ibl_intensity_multiplier;
    }

    // IBL specular
    float3 l_ibl_specular = 0;
    if (p_dfg_texture_srv != RAL_NULL_BINDLESS_INDEX && p_specular_ld_texture_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_ibl_specular = evaluate_ibl_specular(
            p_normal_ws, p_view_dir, p_roughness, p_dfg_texture_srv,
            p_dfg_texture_size, p_specular_ld_texture_srv,
            p_specular_ld_mip_count, p_specular_f0, 1) * l_ibl_intensity_multiplier;
    }

    // Compute specular AO
    float l_nov = saturate(dot(p_normal_ws, p_view_dir));
    float l_alpha = clamp_and_remap_roughness(p_roughness);
    float l_spec_ao = compute_specular_occlusion(l_nov, p_ao, l_alpha);

    // Fake Earth shadow BRDF
    float l_fake_earth_ibl_shadow = fake_earth_ibl_shadow_brdf(p_planet_normal, p_light_list);

    // Apply IBL
    float3 l_ibl = (l_ibl_diffuse * p_ao + l_ibl_specular * l_spec_ao) * l_fake_earth_ibl_shadow;

    return l_ibl;
}

struct barycentric_deriv_t
{
    float3 m_lambda;
    float3 m_ddx;
    float3 m_ddy;
};

// From: http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
barycentric_deriv_t calc_full_bary(float4 p_t0, float4 p_t1, float4 p_t2, float2 p_pixel_ndc, float2 p_win_size, float p_render_scale)
{
    barycentric_deriv_t l_ret = (barycentric_deriv_t)0;

    float3 l_inv_w = rcp(float3(p_t0.w, p_t1.w, p_t2.w));

    float2 l_ndc0 = p_t0.xy * l_inv_w.x;
    float2 l_ndc1 = p_t1.xy * l_inv_w.y;
    float2 l_ndc2 = p_t2.xy * l_inv_w.z;

    float l_inv_det = rcp(determinant(float2x2(l_ndc2 - l_ndc1, l_ndc0 - l_ndc1)));
    l_ret.m_ddx = float3(l_ndc1.y - l_ndc2.y, l_ndc2.y - l_ndc0.y, l_ndc0.y - l_ndc1.y) * l_inv_det * l_inv_w;
    l_ret.m_ddy = float3(l_ndc2.x - l_ndc1.x, l_ndc0.x - l_ndc2.x, l_ndc1.x - l_ndc0.x) * l_inv_det * l_inv_w;
    float l_ddx_sum = dot(l_ret.m_ddx, float3(1, 1, 1));
    float l_ddy_sum = dot(l_ret.m_ddy, float3(1, 1, 1));

    float2 l_delta_vec = p_pixel_ndc - l_ndc0;
    float l_interp_inv_w = l_inv_w.x + l_delta_vec.x * l_ddx_sum + l_delta_vec.y * l_ddy_sum;
    float l_interp_w = rcp(l_interp_inv_w);

    l_ret.m_lambda.x = l_interp_w * (l_inv_w[0] + l_delta_vec.x * l_ret.m_ddx.x + l_delta_vec.y * l_ret.m_ddy.x);
    l_ret.m_lambda.y = l_interp_w * (0.0f       + l_delta_vec.x * l_ret.m_ddx.y + l_delta_vec.y * l_ret.m_ddy.y);
    l_ret.m_lambda.z = l_interp_w * (0.0f       + l_delta_vec.x * l_ret.m_ddx.z + l_delta_vec.y * l_ret.m_ddy.z);

    l_ret.m_ddx *= ((2.0 * p_render_scale) / p_win_size.x);
    l_ret.m_ddy *= ((2.0 * p_render_scale) / p_win_size.y);
    l_ddx_sum   *= ((2.0 * p_render_scale) / p_win_size.x);
    l_ddy_sum   *= ((2.0 * p_render_scale) / p_win_size.y);

    l_ret.m_ddy *= -1.0f;
    l_ddy_sum   *= -1.0f;

    float l_interpW_ddx = 1.0f / (l_interp_inv_w + l_ddx_sum);
    float l_interpW_ddy = 1.0f / (l_interp_inv_w + l_ddy_sum);

    l_ret.m_ddx = l_interpW_ddx * (l_ret.m_lambda * l_interp_inv_w + l_ret.m_ddx) - l_ret.m_lambda;
    l_ret.m_ddy = l_interpW_ddy * (l_ret.m_lambda * l_interp_inv_w + l_ret.m_ddy) - l_ret.m_lambda;

    return l_ret;
}

float3 interpolate_with_deriv(barycentric_deriv_t p_deriv, float p_v0, float p_v1, float p_v2)
{
    float3 l_merged_v = float3(p_v0, p_v1, p_v2);
    float3 l_ret;
    l_ret.x = dot(l_merged_v, p_deriv.m_lambda);
    l_ret.y = dot(l_merged_v, p_deriv.m_ddx);
    l_ret.z = dot(l_merged_v, p_deriv.m_ddy);
    return l_ret;
}

float interpolate(barycentric_deriv_t p_deriv, float p_v0, float p_v1, float p_v2)
{
    float3 l_merged_v_x = float3(p_v0.x, p_v1.x, p_v2.x);
    float l_ret = dot(l_merged_v_x, p_deriv.m_lambda);

    return l_ret;
}

float2 interpolate(barycentric_deriv_t p_deriv, float2 p_v0, float2 p_v1, float2 p_v2)
{
    float3 l_merged_v_x = float3(p_v0.x, p_v1.x, p_v2.x);
    float3 l_merged_v_y = float3(p_v0.y, p_v1.y, p_v2.y);
    float2 l_ret;
    l_ret.x = dot(l_merged_v_x, p_deriv.m_lambda);
    l_ret.y = dot(l_merged_v_y, p_deriv.m_lambda);

    return l_ret;
}

float3 interpolate(barycentric_deriv_t p_deriv, float3 p_v0, float3 p_v1, float3 p_v2)
{
    float3 l_merged_v_x = float3(p_v0.x, p_v1.x, p_v2.x);
    float3 l_merged_v_y = float3(p_v0.y, p_v1.y, p_v2.y);
    float3 l_merged_v_z = float3(p_v0.z, p_v1.z, p_v2.z);
    float3 l_ret;
    l_ret.x = dot(l_merged_v_x, p_deriv.m_lambda);
    l_ret.y = dot(l_merged_v_y, p_deriv.m_lambda);
    l_ret.z = dot(l_merged_v_z, p_deriv.m_lambda);

    return l_ret;
}

float4 interpolate(barycentric_deriv_t p_deriv, float4 p_v0, float4 p_v1, float4 p_v2)
{
    float3 l_merged_v_x = float3(p_v0.x, p_v1.x, p_v2.x);
    float3 l_merged_v_y = float3(p_v0.y, p_v1.y, p_v2.y);
    float3 l_merged_v_z = float3(p_v0.z, p_v1.z, p_v2.z);
    float3 l_merged_v_w = float3(p_v0.w, p_v1.w, p_v2.w);
    float4 l_ret;
    l_ret.x = dot(l_merged_v_x, p_deriv.m_lambda);
    l_ret.y = dot(l_merged_v_y, p_deriv.m_lambda);
    l_ret.z = dot(l_merged_v_z, p_deriv.m_lambda);
    l_ret.w = dot(l_merged_v_w, p_deriv.m_lambda);

    return l_ret;
}

//todo check if these are faster/better than the interpolate function above
float attribute_at_bary(float p_a0, float p_a1, float p_a2, float3 p_bary)
{
    return mad(p_a0, p_bary.z, mad(p_a1, p_bary.x, p_a2 * p_bary.y));
}

float2 attribute_at_bary(float2 p_a0, float2 p_a1, float2 p_a2, float3 p_bary)
{
    return mad(p_a0, p_bary.z, mad(p_a1, p_bary.x, p_a2 * p_bary.y));
}

float3 attribute_at_bary(float3 p_a0, float3 p_a1, float3 p_a2, float3 p_bary)
{
    return mad(p_a0, p_bary.z, mad(p_a1, p_bary.x, p_a2 * p_bary.y));
}

float4 attribute_at_bary(float4 p_a0, float4 p_a1, float4 p_a2, float3 p_bary)
{
    return mad(p_a0, p_bary.z, mad(p_a1, p_bary.x, p_a2 * p_bary.y));
}

static const float2 g_fullscreen_triangle_texcoords[3] =
{
    float2(0.0f, 2.0f),
    float2(0.0f, 0.0f),
    float2(2.0f, 0.0f),
};

static const float2 g_fullscreen_quad_texcoords[6] =
{
    float2(0.0f, 1.0f),
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 0.0f),
    float2(1.0f, 1.0f)
};

static const float3 g_cube_lookup_vectors[6][6] = // [face_id][vertex_id]
{
    // Bottom left,        Top left,           Top right,          Bottom left,        Top right,          Bottom right
    {  float3( 1, -1,  1), float3( 1,  1,  1), float3( 1,  1, -1), float3( 1, -1,  1), float3( 1,  1, -1), float3( 1, -1, -1) }, // Face 0
    {  float3(-1, -1, -1), float3(-1,  1, -1), float3(-1,  1,  1), float3(-1, -1, -1), float3(-1,  1,  1), float3(-1, -1,  1) }, // Face 1
    {  float3(-1,  1,  1), float3(-1,  1, -1), float3( 1,  1, -1), float3(-1,  1,  1), float3( 1,  1, -1), float3( 1,  1,  1) }, // Face 2
    {  float3(-1, -1, -1), float3(-1, -1,  1), float3( 1, -1,  1), float3(-1, -1, -1), float3( 1, -1,  1), float3( 1, -1, -1) }, // Face 3
    {  float3(-1, -1,  1), float3(-1,  1,  1), float3( 1,  1,  1), float3(-1, -1,  1), float3( 1,  1,  1), float3( 1, -1,  1) }, // Face 4
    {  float3( 1, -1, -1), float3( 1,  1, -1), float3(-1,  1, -1), float3( 1, -1, -1), float3(-1,  1, -1), float3(-1, -1, -1) }, // Face 5
};

float2 get_fullscreen_triangle_texcoord(uint p_vertex_id)
{
    return g_fullscreen_triangle_texcoords[p_vertex_id];
}

float4 get_fullscreen_triangle_position(uint p_vertex_id)
{
    return float4(2.0f * g_fullscreen_triangle_texcoords[p_vertex_id].x - 1.0f, 1.0f - 2.0f * g_fullscreen_triangle_texcoords[p_vertex_id].y, 0.0f, 1.0f);
}

float2 get_fullscreen_quad_texcoord(uint p_vertex_id)
{
    return g_fullscreen_quad_texcoords[p_vertex_id];
}

float4 get_fullscreen_quad_position(uint p_vertex_id)
{
    return float4(2.0f * g_fullscreen_quad_texcoords[p_vertex_id].x - 1.0f, 1.0f - 2.0f * g_fullscreen_quad_texcoords[p_vertex_id].y, 0.0f, 1.0f);
}

float3 get_cube_lookup_vector(uint p_face_id, uint p_vertex_id)
{
    return g_cube_lookup_vectors[p_face_id][p_vertex_id];
}

#endif // MB_SHADER_LIGHTING_HLSL
