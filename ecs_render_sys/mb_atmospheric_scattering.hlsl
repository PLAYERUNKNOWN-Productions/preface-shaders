// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_atmospheric_scattering_utils.hlsl"

// Push constants
ConstantBuffer<cb_push_atmospheric_scattering_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 procedural_stars(float3 dir)
{
    // Create mask for star intensity
    float star_low_frequency_mask = fbm_octave(3.0 * dir);
    star_low_frequency_mask *= star_low_frequency_mask;

    // Create star shapes from noises
    float star_mask = 0;
    {
        // This one is simple - just use the [0.8, 1.0] part of simplex noise
        float fbm_noise = simplex_noise_3d(100.0 * dir);
        star_mask += smoothstep(0.8, 1.0, fbm_noise);
    }
    {
        // FBM octave noise is fast, but gives not so good star shapes
        // Use grid to improve the star shape size
        float3 abs_dir = abs(dir);
        float2 uv = (abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z) ? dir.yz / dir.x :
                    (abs_dir.y > abs_dir.x && abs_dir.y > abs_dir.z) ? dir.zx / dir.y :
                                                                       dir.xy / dir.z;
        float quad_tile_mask = abs(cos(200. * uv.x) * cos(200. * uv.y));

        float fbm_noise = fbm_octave(100.0 * dir);
        star_mask += smoothstep(0.8, 0.8001, quad_tile_mask * fbm_noise);
    }

    return star_low_frequency_mask * star_mask;
}

float cloud_elevation_mask(float x, float4 val)
{
    float mask = saturate(min((x - val.x) / val.y, 1.0 - (x - val.z) / (val.w - val.z)));
    // mask = mask * mask * (3.0 - 2.0 * mask); // s-curve
    return mask;
}

float global_cloud_mask(float3 pos, float time)
{
    float2 uv = float2(atan2(pos.x, pos.y) / 3.141592 / 2.0 + 0.5, pos.z * 0.5 + 0.5);
    uv.x += time * 0.0001;
    float noise = bindless_tex2d_sample(g_push_constants.m_cloud_2d_texture_srv, (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], uv).r;
    return noise;
}

float sample_density(float3 pos, bool detail)
{
    float height = length(pos);
    float height_from_surface = height - g_push_constants.m_cloud_earth_radius;
    float cloud_start_height = height_from_surface - g_push_constants.m_cloud_start_height;

    float cloud_gradient_normalized = cloud_start_height / g_push_constants.m_cloud_height;

    // Cloud types elevation mask
    float4 stratus_mask      = float4(1800.0, 1850.0, 2150.0,  2200.0);
    float4 cumulus_mask      = float4(500.0,  2000.0, 5000.0,  6000.0);
    float4 cumulonimbus_mask = float4(500.0,  2000.0, 12000.0, 16000.0);

    float gradient = cloud_elevation_mask(height_from_surface, cumulus_mask);


    float global_mask = global_cloud_mask(normalize(pos), g_push_constants.m_time) * 1.0;
    float4 cloud_noise = bindless_tex3d_sample(g_push_constants.m_cloud_3d_texture_srv,
                                               (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                               pos / 300000.0 + g_push_constants.m_time / 500.0,
                                               0);
    float noise = (1.0 - cloud_noise.r) * (cloud_gradient_normalized + 0.01);
    float sum = global_mask - noise * 0.5;

    if(detail)
    {
        float detail_noise = (1.0 - bindless_tex3d_sample(g_push_constants.m_cloud_3d_texture_srv,
                                                          (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                          pos / 20000.0 + (cloud_noise.rba - 0.5) * 0.0 + g_push_constants.m_time / 100.0,
                                                          0).r) * 0.01;
        sum -= detail_noise;
    }

    return gradient * sum * cloud_gradient_normalized * 0.5;
}

float heney_greenstein_phase(float cos_theta, float g)
{
    return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
}

float beers_powder(float x)
{
    return exp(-x);// * (1.0 - exp(-x * 2.0)) / 0.25;
}

float lightmarch(float3 position, float3 dir_to_light)
{
    const int numStepsLight = 6;
    const float lightAbsorptionTowardSun = 0.15;
    const float darknessThreshold = 0.0;

    float angle = dot(normalize(position), dir_to_light);

    float step_size = g_push_constants.m_cloud_height / numStepsLight / angle * 0.5;
    float totalDensity = 0.0;

    for (int i = 0; i < numStepsLight; i ++)
    {
        float3 sampleDir = dir_to_light;;
        position += sampleDir * step_size;
        totalDensity += max(0.0, sample_density(position, false) * step_size);
    }

    float transmittance = beers_powder(totalDensity * lightAbsorptionTowardSun);
    return darknessThreshold + transmittance * (1.0 - darknessThreshold);
}

// Ray-sphere intersection function
//  TODO: Merge with ray_sphere_intersect
bool ray_sphere(float3 ray_origin, float3 ray_dir, float3 sphere_center, float sphere_radius, out float t_front, out float t_back)
{
    float3 oc = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0)
    {
        t_front = -1.0;
        t_back = -1.0;
        return false; // No intersection
    }

    float sqrtD = sqrt(discriminant);
    float t0 = (-b - sqrtD) / (2.0 * a);
    float t1 = (-b + sqrtD) / (2.0 * a);

    // Ensure t0 <= t1
    if (t0 > t1)
    {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    t_front = t0;
    t_back  = t1;

    // Return true if at least one intersection is in front of the ray origin
    return (t0 >= 0.0 || t1 >= 0.0);
}

float3 binary_search_volume(float3 pos, float3 dir)
{
    // first step back outside volume
    pos -= dir;
    float step_size = 1.0;

    for (int i = 0; i < 4; i ++)
    {
        // Check volume
        float density = sample_density(pos, false);

        // if inside volume, step back
        if (density > 0.00002)
        {
            step_size = abs(step_size) * -0.5;
        }
        // if outside volume, step forward
        else
        {
            step_size = abs(step_size) * 0.5;
        }

        pos += dir * step_size;
    }

    return pos;
}

float get_altitude(float3 position)
{
    return length(position - g_push_constants.m_planet_center) - g_push_constants.m_planet_radius;
}

float3 apply_fog(float3 input_color, float3 fog_color, float fog_amount)
{
    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    float camera_altitude = max(0.001, get_altitude(camera.m_camera_pos));
    fog_amount *= 1 - saturate((camera_altitude - g_push_constants.m_fog_camera_height_fadeout_start) * g_push_constants.m_fog_camera_height_fadeout_inv_distance);

    fog_amount = clamp(0, g_push_constants.m_fog_max_amount, fog_amount);

    // HACK: Use the sun position to fade out fog, as else it will glow in the dark. This should use the scattering LUT instead!
    if (g_push_constants.m_fog_sun_fade_enabled)
    {
        float3 planet_center_to_position = normalize(camera.m_camera_pos - g_push_constants.m_planet_center);
        float cos_zenith = dot(planet_center_to_position, g_push_constants.m_sun_light_dir);

        // Remap so the fadeout starts close to when the sun is on the horizon
        fog_amount *= smoothstep(0, 0.3, cos_zenith);
    }

    return lerp(input_color, fog_color, fog_amount);
}

[numthreads(ATMOSPHERIC_SCATTERING_THREAD_GROUP_SIZE, ATMOSPHERIC_SCATTERING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint2 dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(dispatch_thread_id >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    // Get camera cb
    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get uv
    float2 uv = (dispatch_thread_id + 0.5f) / (float2)g_push_constants.m_dst_resolution;

    // Get depth
    float depth = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], uv).r;

    // Get world space local position
    float3 pos_ws_local = get_world_space_local_position(uv, depth, camera.m_inv_view_proj_local);

    // Get initial attributes of the ray
    float3 ray_start = camera.m_camera_pos;
    float3 ray_dir = normalize(pos_ws_local);
    float ray_length = length(pos_ws_local);

    // Calculate inscattering
    float3 light_inscattering = 0;
    float3 light_extinction = 0;
    compute_inscattering_along_ray(ray_start,
                                   ray_dir,
                                   ray_length,
                                   g_push_constants,
                                   light_inscattering,
                                   light_extinction);

    RWTexture2D<float4> output_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];
    float3 lighting = unpack_lighting(output_rt[dispatch_thread_id].rgb);

    // Final scattering result
    // [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 9
    lighting = lighting * light_extinction + light_inscattering;

    // Drawn only to background
    if (depth == 0)
    {
        // Sun disk
        lighting += get_sun_disc_mask(g_push_constants.m_sun_light_dir, ray_dir) * g_push_constants.m_sun_light_color * light_extinction;

        // Stars
        lighting += procedural_stars(ray_dir) * g_push_constants.m_star_intensity * light_extinction;

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
        float4 proj_pos_curr = mul(float4(pos_ws_local, 1.0f), camera.m_view_proj_local);
        float4 proj_pos_prev = mul(float4(pos_ws_local, 1.0f), camera.m_view_proj_local_prev);

        RWTexture2D<float2> velocity_rt = ResourceDescriptorHeap[g_push_constants.m_dst_velocity_uav];
        velocity_rt[dispatch_thread_id.xy] = get_motion_vector_without_jitter(  float2(camera.m_resolution_x, camera.m_resolution_y),
                                                                                proj_pos_curr.xyw,
                                                                                proj_pos_prev.xyw,
                                                                                camera.m_jitter, camera.m_jitter_prev);
#endif
    }
    else
    {
        // Height fog
        if (g_push_constants.m_fog_height_enabled)
        {
            // Compute altitude from ground (roughly)
            float position_altitude = get_altitude(camera.m_camera_pos + pos_ws_local);
            position_altitude -= g_push_constants.m_fog_height_start_from;

            float height_fog_amount = saturate(exp(-position_altitude * g_push_constants.m_fog_height_density));

            // Fade out height fog close to the camera, so visibility is kept
            float z_vs = get_view_depth_from_depth(depth, camera.m_z_near, camera.m_z_far);
            height_fog_amount *= saturate(1 - exp(-z_vs * g_push_constants.m_fog_height_distance_fadeout_density));

            lighting = apply_fog(lighting, g_push_constants.m_fog_height_color, height_fog_amount);
        }

        // Depth fog
        if (g_push_constants.m_fog_depth_enabled)
        {
            float z_vs = get_view_depth_from_depth(depth, camera.m_z_near, camera.m_z_far);
            z_vs -= g_push_constants.m_fog_depth_start_from;

            float depth_fog_amount = saturate(1 - exp(-z_vs * g_push_constants.m_fog_depth_density));
            lighting = apply_fog(lighting, g_push_constants.m_fog_depth_color, depth_fog_amount);
        }
    }


    //// VOLUMETRIC CLOUDS ////
    if (g_push_constants.m_cloud_enabled)
    {
        float3 dir_to_light = g_push_constants.m_sun_light_dir;

        float g = 0.9;
        float cos_theta = dot(normalize(ray_dir), dir_to_light);
        float phase_val = heney_greenstein_phase(cos_theta, g);

        const float light_absorption_through_cloud = 0.15;

        // Ray-sphere intersection to optimize sampling
        float cloud_sphere_radius = g_push_constants.m_cloud_earth_radius + g_push_constants.m_cloud_start_height;
        float cloud_top_sphere_radius = g_push_constants.m_cloud_earth_radius + g_push_constants.m_cloud_start_height + g_push_constants.m_cloud_height;
        float t_base_front, t_base_back, t_top_front, t_top_back;

        // Check intersections for the cloud base and top
        bool hit_cloud_base = ray_sphere(ray_start, ray_dir, float3(0.0, 0.0, 0.0), cloud_sphere_radius, t_base_front, t_base_back);
        bool hit_cloud_top =  ray_sphere(ray_start, ray_dir, float3(0.0, 0.0, 0.0), cloud_top_sphere_radius, t_top_front, t_top_back);

        float player_distance_from_surface = length(ray_start) - g_push_constants.m_cloud_earth_radius;

        bool under_clouds = player_distance_from_surface < g_push_constants.m_cloud_start_height;
        bool inside_clouds = player_distance_from_surface >= g_push_constants.m_cloud_start_height && player_distance_from_surface <= (g_push_constants.m_cloud_start_height + g_push_constants.m_cloud_height);
        bool above_clouds = player_distance_from_surface > (g_push_constants.m_cloud_start_height + g_push_constants.m_cloud_height);

        // Total distance trough cloud layer
        float sample_dist = 0.0;
        // Position where ray starts
        float3 start_pos = ray_start;
        // Distance from cam to ray start
        float start_dist = 0.0;

        // Under clouds
        if (under_clouds)
        {
            start_pos += t_base_back * ray_dir;
            start_dist = t_base_back;
            sample_dist = t_top_back - t_base_back;
        }
        // Above clouds
        else if (above_clouds && hit_cloud_top)
        {
            start_pos += t_top_front * ray_dir;
            start_dist = t_top_front;
            if (hit_cloud_base)
            {
                sample_dist = t_base_front - t_top_front;
            }
            else
            {
                sample_dist = t_top_back - t_top_front;
            }
        }
        // In between cloud layers
        else if (inside_clouds)
        {
            float maxSampleDist = 200000.0;
            // small isue here is that ray can re-enter cloud layer again, thats not taken into account
            if (hit_cloud_base)
            {
                sample_dist = min(t_base_front, maxSampleDist);
            }
            else
            {
                sample_dist = min(t_top_back, maxSampleDist);
            }
        }

        // Maximum length ray can travel in cloud area
        float maximum_cloud_ray_length = sqrt(cloud_top_sphere_radius*cloud_top_sphere_radius - cloud_sphere_radius*cloud_sphere_radius) * 2.0;
        float ray_fraction = sample_dist / maximum_cloud_ray_length;

        int steps = 64;
        // 64-128 steps based on ray length
        steps = lerp(64, 128, ray_fraction / (start_dist + 1.0));
        // steps based on travel dist
        // steps = int(floor((sample_dist) / 300.0));

        float step_size = sample_dist / steps;
        float step_size_inside_volume = step_size * 0.5;
        float step_size_outside_volume = step_size * 1.5;
        step_size = step_size_outside_volume;


        float3 sample_pos = start_pos;
        float sum_distance = 0.0;

        bool has_entered_volume = false;

        // March through volume:
        float transmittance = 1.0;
        float3 light_energy = 0.0;

        for(int i = 0; i < steps; i++)
        {
            float density = sample_density(sample_pos, has_entered_volume);

            if (density > 0.00002)
            {
                // decrease the stepsize if you have entered cloud // we decrease noise costs when we are not yet in the volume
                if (has_entered_volume == false)
                {
                    // step back from volume to check again with smaller steps
                    sample_pos -= ray_dir * step_size_outside_volume * 0.5;
                    // sample_pos = binary_search_volume(sample_pos, ray_dir * step_size_outside_volume);

                    // remove one from the loop as we are tracing back
                    i -= 1;

                    // Make stepsize smaller
                    step_size = step_size_inside_volume;
                    has_entered_volume = true;
                    continue;
                }

                float light_transmittance = lightmarch(sample_pos, dir_to_light);
                light_energy += density * step_size * transmittance * light_transmittance * phase_val;
                transmittance *= exp(-density * step_size * light_absorption_through_cloud);

                sum_distance += step_size;

                // Exit early if T is close to zero as further samples won't affect the result much
                // Exit if
                if (transmittance < 0.01 || sum_distance > sample_dist)
                {
                    // transmittance /= 1.5; // this removes some of the banding for some reason // compared value * 200.0
                    break;
                }
            }
            // If you no longer in volume, but has_entered_volume is still true, this means you just exited the volume
            else if(has_entered_volume == true)
            {
                step_size = step_size_outside_volume;
                has_entered_volume = false;
            }
            sample_pos += ray_dir * step_size;

            // If sample is inside planet or asset, stop sampling
            // if (length(pos_ws_local) < distance(ray_start, sample_pos))
            if (dot(pos_ws_local,pos_ws_local) < dot(ray_start - sample_pos, ray_start - sample_pos))
            {
                break;
            }
        }

        float3 ambient_light = float3(0.3, 0.5, 1.0) * g_push_constants.m_sun_light_color * 0.05;
        lighting = lerp((light_energy * g_push_constants.m_sun_light_color + ambient_light) * saturate(dot(normalize(sample_pos), dir_to_light)), lighting, transmittance);
    }

    output_rt[dispatch_thread_id] = float4(pack_lighting(lighting), 1);
}
