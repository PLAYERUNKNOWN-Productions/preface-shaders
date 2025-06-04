// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

// Push constants
ConstantBuffer<cb_push_raytracing_combine_t> g_push_constants  : register(REGISTER_PUSH_CONSTANTS);

[numthreads(RAYTRACING_DENOISING_THREAD_GROUP_SIZE, RAYTRACING_DENOISING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint2 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip pixels outside of the tile
    if (p_dispatch_thread_id.x >= g_push_constants.m_dst_resolution_x ||
        p_dispatch_thread_id.y >= g_push_constants.m_dst_resolution_y)
    {
        return;
    }

    // Get resources
    RWTexture2D<float4> l_texure_dst                            = ResourceDescriptorHeap[g_push_constants.m_texture_dst_uav];
    RWTexture2D<float4> l_raytracing_direct                     = ResourceDescriptorHeap[g_push_constants.m_direct_lighting_uav];
    RWTexture2D<float4> l_raytracing_indirect_diffuse           = ResourceDescriptorHeap[g_push_constants.m_indirect_diffuse_uav];
    RWTexture2D<float4> l_raytracing_indirect_specular          = ResourceDescriptorHeap[g_push_constants.m_indirect_specular_uav];
    RWTexture2D<float4> l_raytracing_denoised_indirect_diffuse  = ResourceDescriptorHeap[g_push_constants.m_denoised_indirect_diffuse_uav];
    RWTexture2D<float4> l_raytracing_denoised_indirect_specular = ResourceDescriptorHeap[g_push_constants.m_denoised_indirect_specular_uav];
    RWTexture2D<float4> l_raytracing_acc                        = ResourceDescriptorHeap[g_push_constants.m_accumulated_lighting_uav];
    RWTexture2D<float4> l_diffuse_reflectance_rt_uav            = ResourceDescriptorHeap[g_push_constants.m_raytracing_diffuse_reflectance_rt_uav];
    RWTexture2D<float4> l_depth_rt_uav                          = ResourceDescriptorHeap[g_push_constants.m_raytracing_depth_rt_uav];
    RWTexture2D<float4> l_normals_rt_uav                        = ResourceDescriptorHeap[g_push_constants.m_raytracing_normals_rt_uav];

    float3 l_denoised_indirect_diffuse = l_raytracing_denoised_indirect_diffuse[p_dispatch_thread_id].xyz;
    float3 l_denoised_indirect_specular = l_raytracing_denoised_indirect_specular[p_dispatch_thread_id].xyz;
    float3 l_indirect_diffuse = l_raytracing_indirect_diffuse[p_dispatch_thread_id].xyz;
    float3 l_indirect_specular = l_raytracing_indirect_specular[p_dispatch_thread_id].xyz;
    float3 l_direct_lighting = l_raytracing_direct[p_dispatch_thread_id].xyz;
    float3 l_diffuse_reflectance = l_diffuse_reflectance_rt_uav[p_dispatch_thread_id].xyz;
    float4 l_pos_depth = l_depth_rt_uav[p_dispatch_thread_id];
    float3 l_normals = l_normals_rt_uav[p_dispatch_thread_id].xyz;

    // Select output
    float3 l_lighting = 0;
    if(g_push_constants.m_output_type == 0)
    {
        // We are applying accumulation for diffuse reflectance to get AA
        float3 l_diffuse_reflectance_avg = l_diffuse_reflectance  / (float)(g_push_constants.m_acc_frame_index + 1.0f);
        l_lighting = l_direct_lighting + l_diffuse_reflectance_avg * l_denoised_indirect_diffuse + l_denoised_indirect_specular;
    }
    else if(g_push_constants.m_output_type == 1)
    {
        l_lighting = l_denoised_indirect_diffuse;
    }
    else if(g_push_constants.m_output_type == 2)
    {
        l_lighting = l_denoised_indirect_specular;
    }else
    {
        l_lighting = l_raytracing_acc[p_dispatch_thread_id].xyz;
    }

    //l_texure_dst[p_dispatch_thread_id] = float4(l_normals, 0);
    l_texure_dst[p_dispatch_thread_id] = float4(pack_lighting(l_lighting / (float)(g_push_constants.m_acc_frame_index + 1.0f)), 1.0f);
    //l_texure_dst[p_dispatch_thread_id] = float4(l_pos_depth.xyz / 100.0, 0);

#if 0
    // Reccurent blur
    // Supply denoised image input as the raytracing accumulation texture
    l_raytracing_indirect_diffuse[p_dispatch_thread_id] = l_raytracing_denoised_indirect_diffuse[p_dispatch_thread_id];
#endif
}
