// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

// Root constants
ConstantBuffer<cb_push_resolve_gsm_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//#define DEBUG_RAY_DIR
#define REVERSE_Z

float load_depth(uint2 p_pixel_coord, Texture2D<float> p_depth_buffer)
{
#ifdef REVERSE_Z
    return 1.0f - p_depth_buffer.Load(uint3(p_pixel_coord, 0));
#else
    return p_depth_buffer.Load(uint3(p_pixel_coord, 0));
#endif
}

[numthreads(MB_GSM_THREADGROUP_SIZE_X, MB_GSM_THREADGROUP_SIZE_Y, 1)]
void cs_main(uint2 p_dispatch_thread_id : SV_DispatchThreadID)
{
    RWTexture2D<float> l_resolved_gsm_texture = ResourceDescriptorHeap[g_push_constants.m_resolved_texture_uav];
    Texture2D<float> l_global_shadow_map = ResourceDescriptorHeap[g_push_constants.m_gsm_srv];

    float2 l_pixel_coord = p_dispatch_thread_id + 0.5f;
    float l_start_depth_value = load_depth(l_pixel_coord, l_global_shadow_map);

#ifdef DEBUG_RAY_DIR
    bool l_output_debug_ray = all(g_push_constants.m_mouse_pos == p_dispatch_thread_id) &&
                              g_push_constants.m_debug_texture_uav != RAL_NULL_BINDLESS_INDEX;
#endif

    float l_depth_range = g_push_constants.m_gsm_z_far - g_push_constants.m_gsm_z_near;
    float l_depth_coord = l_start_depth_value * l_depth_range + g_push_constants.m_gsm_z_near;
    float3 l_march_dir = g_push_constants.m_march_dir;

    float3 l_march_pos = float3(l_pixel_coord, l_depth_coord);
    while(all(l_march_pos >= float3(0, 0, g_push_constants.m_gsm_z_near)) &&
          all(l_march_pos < float3(g_push_constants.m_gsm_dimensions, g_push_constants.m_gsm_z_far)))
    {
        l_march_pos += l_march_dir;
        float l_terrain_depth_ndc = load_depth(l_march_pos.xy, l_global_shadow_map);
        l_terrain_depth_ndc += 0.0000001f;//bias
        float l_terrain_depth_pixel_coord = l_terrain_depth_ndc * l_depth_range + g_push_constants.m_gsm_z_near;

#ifdef DEBUG_RAY_DIR
        if(l_output_debug_ray)
        {
            RWTexture2D<float> l_gsm_debug_texture = ResourceDescriptorHeap[g_push_constants.m_debug_texture_uav];
            l_gsm_debug_texture[l_march_pos.xy] = 1;
        }
#endif

        if(l_march_pos.z > l_terrain_depth_pixel_coord)
        {
            l_resolved_gsm_texture[p_dispatch_thread_id] = 0;
            return;
        }
    }

    l_resolved_gsm_texture[p_dispatch_thread_id] = 1;
}
