// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "mb_lighting_common.hlsl"
#include "mb_atmospheric_scattering_utils.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_atmospheric_scattering_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
float3 procedural_stars(float3 p_dir)
{
    // Create mask for star intensity
    float l_star_low_frequency_mask = fbm_octave(3.0 * p_dir);
    l_star_low_frequency_mask *= l_star_low_frequency_mask;

    // Create star shapes from noises
    float l_star_mask = 0;
    {
        // This one is simple - just use the [0.8, 1.0] part of simplex noise
        float l_fbm_noise = simplex_noise_3d(100.0 * p_dir);
        l_star_mask += smoothstep(0.8, 1.0, l_fbm_noise);
    }
    {
        // FBM octave noise is fast, but gives not so good star shapes
        // Use grid to improve the star shape size
        float3 l_abs_dir = abs(p_dir);
        float2 l_uv = (l_abs_dir.x > l_abs_dir.y && l_abs_dir.x > l_abs_dir.z) ? p_dir.yz / p_dir.x :
                      (l_abs_dir.y > l_abs_dir.x && l_abs_dir.y > l_abs_dir.z) ? p_dir.zx / p_dir.y :
                                                                                 p_dir.xy / p_dir.z;
        float l_quad_tile_mask = abs(cos(200. * l_uv.x) * cos(200. * l_uv.y));

        float l_fbm_noise = fbm_octave(100.0 * p_dir);
        l_star_mask += smoothstep(0.8, 0.8001, l_quad_tile_mask * l_fbm_noise);
    }

    return l_star_low_frequency_mask * l_star_mask;
}

//-----------------------------------------------------------------------------
[numthreads(ATMOSPHERIC_SCATTERING_THREAD_GROUP_SIZE, ATMOSPHERIC_SCATTERING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];

    // Get camera cb
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get uv
    float2 l_uv = (p_dispatch_thread_id.xy + 0.5f) / (float2)g_push_constants.m_dst_resolution;

    // Get remapped uv
    float2 l_remapped_uv = get_remapped_uv(l_uv, l_camera.m_render_scale);

    // Get depth
    float l_depth = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_remapped_uv).r;

    // Get world space local position
    float3 l_pos_ws_local = get_world_space_local_position(l_uv, l_depth, l_camera.m_inv_view_proj_local);

    // Get initial attributes of the ray
    float3 l_ray_start = l_camera.m_camera_pos;
    float3 l_ray_dir = normalize(l_pos_ws_local);
    float l_ray_length = length(l_pos_ws_local);

    // Calculate inscattering
    float3 l_light_inscattering = 0;
    float3 l_light_extinction = 0;
    compute_inscattering_along_ray(l_ray_start,
                                    l_ray_dir,
                                    l_ray_length,
                                    g_push_constants,
                                    l_light_inscattering,
                                    l_light_extinction);

    float3 l_lighting = unpack_lighting(l_rt[p_dispatch_thread_id.xy].rgb);

    // Final scattering result
    // [Nishita 1993, Display of The Earth Taking into Account Atmospheric Scattering] : Equation 9
    l_lighting = l_lighting * l_light_extinction + l_light_inscattering;

    // Sun disk(only when is not occluded by depth)
    l_lighting += (l_depth == 0) * get_sun_disc_mask(g_push_constants.m_sun_light_dir, l_ray_dir) * g_push_constants.m_sun_light_color * l_light_extinction;

    // Stars
    l_lighting += (l_depth == 0) * procedural_stars(l_ray_dir) * g_push_constants.m_star_intensity * l_light_extinction;

    l_rt[p_dispatch_thread_id.xy].rgb = pack_lighting(l_lighting);
}
