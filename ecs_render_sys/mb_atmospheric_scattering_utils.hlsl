// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MBSHADER_ATMOSPHERIC_SCATTERING_UTILS
#define MBSHADER_ATMOSPHERIC_SCATTERING_UTILS

#define FULLY_REALTIME (0)

//-----------------------------------------------------------------------------
// Return the scale of the ray direction (near, far)
float2 ray_sphere_intersect(float3 p_ray_origin, float3 p_ray_dir,
                            float3 p_sphere_center, float p_sphere_radius)
{
    p_ray_origin -= p_sphere_center;
    float l_a = dot(p_ray_dir, p_ray_dir);
    float l_b = 2.0f * dot(p_ray_origin, p_ray_dir);
    float l_c = dot(p_ray_origin, p_ray_origin) - p_sphere_radius * p_sphere_radius;
    float l_d = l_b * l_b - 4.0f * l_a * l_c;
    if (l_d < 0)
    {
        return -1;
    }
    else
    {
        l_d = sqrt(l_d);
        return float2(-l_b - l_d, -l_b + l_d) / (2.0f * l_a);
    }
}

//-----------------------------------------------------------------------------
// Density ratio
// [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 2
float2 get_density_ratio(float3 p_position, float3 p_planet_center, float p_planet_radius, float2 p_density_scale_height)
{
    float l_altitude = length(p_position - p_planet_center) - p_planet_radius;

    // If altitude is under sea level, always use density ratio at sea level
    l_altitude = max(l_altitude, 0.0f);

    return exp(-l_altitude.xx / p_density_scale_height);
}

//-----------------------------------------------------------------------------
// Get optical depth along the light direction
float2 get_optical_depth_along_light_direction(float3 p_ray_start, float3 p_ray_dir, float3 p_planet_center, float p_planet_radius,
                                               float p_atmosphere_height, float2 p_density_scale_height, uint p_sample_count)
{
    // Get the intersection with the planet
    float2 l_intersection = ray_sphere_intersect(p_ray_start, p_ray_dir, p_planet_center, p_planet_radius);

    // Intersect with the planet
    if (l_intersection.x > 0)
    {
        return 1e20;
    }

    // Get the intersection with the outer atmosphere
    l_intersection = ray_sphere_intersect(p_ray_start, p_ray_dir, p_planet_center, p_planet_radius + p_atmosphere_height);

    float l_ray_end = l_intersection.y;

    // Compute the optical depth along the ray
    float l_step_size = l_ray_end / p_sample_count;
    float2 l_optical_depth = 0.0f;
    for (uint l_i = 0; l_i < p_sample_count; l_i++)
    {
        // Sample in the middle of the segment
        float3 l_position = p_ray_start + (l_i + 0.5f) * l_step_size * p_ray_dir;

        float2 l_density_ratio = get_density_ratio(l_position, p_planet_center, p_planet_radius, p_density_scale_height);

        l_optical_depth += l_density_ratio * l_step_size;
    }

    return l_optical_depth;
}

//-----------------------------------------------------------------------------
// Phase function for Rayleigh scattering
// [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 5
// We divided by an extra 4 pi since we want to use the scattering coefficient in the final scattering equation
float phase_function_rayleigh(float p_cos_theta)
{
    return (3.0f / (16.0f * M_PI)) * (1 + p_cos_theta * p_cos_theta);
}

//-----------------------------------------------------------------------------
// Phase function for Mie scattering
// Cornette-Shanks phase function
// [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 5
// We divided by an extra 4 pi since we want to use the scattering coefficient in the final scattering equation
float phase_function_mie(float p_cos_theta, float p_mie_g)
{
    float l_mie_g2 = p_mie_g * p_mie_g;
    return 1.5f * 1.0f / (4.0f * M_PI)
           * (1.0f - l_mie_g2) * (1.0f + p_cos_theta * p_cos_theta)
           * pow(1.0f + l_mie_g2 - 2.0f * p_mie_g * p_cos_theta, -3.0f / 2.0f) / (2.0f + l_mie_g2);
}

//-----------------------------------------------------------------------------
// [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 8
// Scattering coefficient and phase function are excluded here
void compute_inscattering_at_point(float2 p_density_ratio_at_point, float2 p_optical_depth_camera_to_point, float2 p_optical_depth_point_to_sun,
                                   float3 p_extinction_rayleigh, float3 p_extinction_mie,
                                   out float3 p_inscattering_rayleigh, out float3 p_inscattering_mie)
{
    float2 l_optical_depth = p_optical_depth_camera_to_point + p_optical_depth_point_to_sun;

    float3 l_transmittance_rayleigh = l_optical_depth.x * p_extinction_rayleigh;
    float3 l_transmittance_mie = l_optical_depth.y * p_extinction_mie;

    float3 l_extinction = exp(-(l_transmittance_rayleigh + l_transmittance_mie));

    p_inscattering_rayleigh = p_density_ratio_at_point.x * l_extinction;
    p_inscattering_mie = p_density_ratio_at_point.y * l_extinction;
}

//-----------------------------------------------------------------------------
// Sun disk function
float get_sun_disc_mask(float3 p_light_dir, float3 p_ray_dir)
{
    float l_sun_angular_size = 0.00872664626f;
    float l_sun_cos = cos(l_sun_angular_size);
    float l_sun_disk = smoothstep(l_sun_cos * 0.99999f, l_sun_cos, dot(p_light_dir, p_ray_dir));
    //float l_sun_disk = dot(p_light_dir, p_ray_dir) > l_sun_cos;
    return l_sun_disk;
}

//-----------------------------------------------------------------------------
void compute_inscattering_along_ray(float3 p_ray_start,
                                    float3 p_ray_dir,
                                    float p_ray_length,
                                    cb_push_atmospheric_scattering_t p_push_constants,
                                    out float3 p_light_inscattering,
                                    out float3 p_light_extinction)
{
    // Get intersections with the outer atmosphere (near, far)
    float2 l_intersections = ray_sphere_intersect(p_ray_start, p_ray_dir, p_push_constants.m_planet_center, p_push_constants.m_planet_radius + p_push_constants.m_atmosphere_height);

    // Get starting point and end point of the ray
    float p_ray_start_scale = max(l_intersections.x, 0.0f);
    float l_ray_end_scale = min(l_intersections.y, p_ray_length);

    if (p_ray_start_scale >= p_ray_length ||    // Ray occluded
        l_ray_end_scale < 0)                    // No intersection with the atmosphere
    {
        p_light_inscattering = 0.0f;
        p_light_extinction = 1.0f;
        return;
    }

    // Update the starting position of the ray
    p_ray_start = p_ray_start + p_ray_dir * p_ray_start_scale;

    // Integrate in-scattering
    float2 l_optical_depth_camera_to_point = 0;
    float3 l_integrated_inscattering_rayleigh = 0;
    float3 l_integrated_inscattering_mie = 0;
    float l_step_size = (l_ray_end_scale - p_ray_start_scale) / p_push_constants.m_sample_count_view_direction;
    for (uint l_i = 1; l_i <= p_push_constants.m_sample_count_view_direction; l_i++)
    {
        float3 l_position = p_ray_start + l_i * l_step_size * p_ray_dir;

#if FULLY_REALTIME
        // Get density ratio
        float2 l_density_ratio = get_density_ratio(l_position, p_push_constants.m_planet_center, p_push_constants.m_planet_radius, p_push_constants.m_density_scale_height);

        // Integrate the optical depth along the view direction
        l_optical_depth_camera_to_point += l_density_ratio * l_step_size;

        // Get optical depth along the light direction
        float2 l_optical_depth_point_to_sun = get_optical_depth_along_light_direction(l_position, p_push_constants.m_sun_light_dir, p_push_constants.m_planet_center, p_push_constants.m_planet_radius,
                                                                                      p_push_constants.m_atmosphere_height, p_push_constants.m_density_scale_height, p_push_constants.m_sample_count_light_direction);
#else
        // UV for the lookup table
        float3 l_planet_center_to_position = l_position - p_push_constants.m_planet_center;
        float l_altitude = length(l_planet_center_to_position) - p_push_constants.m_planet_radius;
        float l_cos_zenith = dot(normalize(l_planet_center_to_position), p_push_constants.m_sun_light_dir);
        float2 l_lut_uv = float2(l_altitude / p_push_constants.m_atmosphere_height, (l_cos_zenith + 1.0f) * 0.5f);

        float4 l_lut_val = bindless_tex2d_sample_level(p_push_constants.m_lut_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_lut_uv);

        // Get density ratio
        float2 l_density_ratio = l_lut_val.xy;

        // Integrate the optical depth along the view direction
        l_optical_depth_camera_to_point += l_density_ratio * l_step_size;

        // Get optical depth along the light direction
        float2 l_optical_depth_point_to_sun = l_lut_val.zw;
#endif

        // Get in-scattering
        float3 l_inscattering_rayleigh = 0;
        float3 l_inscattering_mie = 0;
        compute_inscattering_at_point(l_density_ratio, l_optical_depth_camera_to_point, l_optical_depth_point_to_sun,
                                        p_push_constants.m_rayleigh_scattering_coefficient, p_push_constants.m_mie_scattering_coefficient,
                                        l_inscattering_rayleigh, l_inscattering_mie);

        // Integrate the in-scattering
        l_integrated_inscattering_rayleigh += l_inscattering_rayleigh * l_step_size;
        l_integrated_inscattering_mie += l_inscattering_mie * l_step_size;
    }

    // Apply phase function
    float l_cos_theta = dot(p_ray_dir, p_push_constants.m_sun_light_dir);
    l_integrated_inscattering_rayleigh *= phase_function_rayleigh(l_cos_theta);
    l_integrated_inscattering_mie *= phase_function_mie(l_cos_theta, p_push_constants.m_mie_g);

    // On-off
    l_integrated_inscattering_rayleigh *= (float)p_push_constants.m_enable_rayleigh_scattering;
    l_integrated_inscattering_mie *= (float)p_push_constants.m_enable_mie_scattering;

    // Calculate the in-scattering
    p_light_inscattering = (l_integrated_inscattering_rayleigh * p_push_constants.m_rayleigh_scattering_coefficient + l_integrated_inscattering_mie * p_push_constants.m_mie_scattering_coefficient)
                           * p_push_constants.m_solar_irradiance * p_push_constants.m_sun_light_color;

    // Calculate the out-scattering (without reflected light from earth)
    p_light_extinction = exp(-(l_optical_depth_camera_to_point.x * p_push_constants.m_rayleigh_scattering_coefficient + l_optical_depth_camera_to_point.y * p_push_constants.m_mie_scattering_coefficient));
}

#endif // MBSHADER_ATMOSPHERIC_SCATTERING_UTILS
