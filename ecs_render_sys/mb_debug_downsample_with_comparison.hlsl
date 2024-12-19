// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_downsample_with_comparison_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(DOWNSAMPLE_THREAD_GROUP_SIZE, DOWNSAMPLE_THREAD_GROUP_SIZE, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    if (any(p_dispatch_thread_id.xy >= g_push_constants.m_dst_resolution))
    {
        return;
    }

#if SINGLE_CHANNEL_UINT
    RWTexture2D<uint> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];
#elif SINGLE_CHANNEL_FLOAT
    RWTexture2D<float> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];
#elif LIGHTING
    RWTexture2D<float4> l_rt = ResourceDescriptorHeap[g_push_constants.m_dst_texture_uav];
#endif

    const uint2 l_pos = p_dispatch_thread_id.xy;
    float2 l_base_uv = float2(2.0f * p_dispatch_thread_id.x * g_push_constants.m_inv_src_resolution.x,
                              2.0f * p_dispatch_thread_id.y * g_push_constants.m_inv_src_resolution.y); // Upper left corner of the sampling area

    // Check the number of steps needed for the sampling loop.  *0 means can't be precomputed
    uint2 l_step;
    l_step.x = (g_push_constants.m_precomputed_step.x != 0) ? g_push_constants.m_precomputed_step.x : ((p_dispatch_thread_id.x == g_push_constants.m_dst_resolution.x - 1) ? 3 : 2);
    l_step.y = (g_push_constants.m_precomputed_step.y != 0) ? g_push_constants.m_precomputed_step.y : ((p_dispatch_thread_id.y == g_push_constants.m_dst_resolution.y - 1) ? 3 : 2);

#if SINGLE_CHANNEL_UINT
    Texture2D<uint> l_texture = ResourceDescriptorHeap[g_push_constants.m_src_texture_srv];
#endif

#if SINGLE_CHANNEL_UINT

#if LOWER
    uint l_retval = 0xFFFFFFFF;
#elif HIGHER
    uint l_retval = 0;
#endif

#elif SINGLE_CHANNEL_FLOAT

#if LOWER
    float l_color = FLT_MAX;
#elif HIGHER
    float l_color = -FLT_MAX;
#endif

#elif LIGHTING

#if LOWER
    float4 l_color = float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
#if LIGHTING
    float l_lum = FLT_MAX;
#endif
#elif HIGHER
    float4 l_color = float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
#if LIGHTING
    float l_lum = -FLT_MAX;
#endif
#endif // HIGHER

#endif // LIGHTING

    // If 2x2, take advantage of the bilinear filter
    if (all(l_step == 2))
    {
#if SINGLE_CHANNEL_UINT
        uint3 l_base_location = uint3(p_dispatch_thread_id.x << 1, p_dispatch_thread_id.y << 1, 0);
#elif SINGLE_CHANNEL_FLOAT || LIGHTING
        l_base_uv = l_base_uv + g_push_constants.m_inv_src_resolution * 0.5f;
#endif
        [unroll(4)]
        for (uint l_x = 0; l_x < 2; l_x++)
        {
            for (uint l_y = 0; l_y < 2; l_y++)
            {
#if SINGLE_CHANNEL_UINT
                uint3 l_location = l_base_location + uint3(l_x, l_y, 0);
                uint l_val = l_texture.Load(l_location);

#if LOWER
                l_retval = min(l_retval, l_val);
#elif HIGHER
                l_retval = max(l_retval, l_val);
#endif

#elif SINGLE_CHANNEL_FLOAT
                float2 l_uv = l_base_uv + g_push_constants.m_inv_src_resolution * float2(l_x, l_y);
                float l_val = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).r;

#if LOWER
                l_color = min(l_color, l_val);
#elif HIGHER
                l_color = max(l_color, l_val);
#endif

#elif LIGHTING
                float2 l_uv = l_base_uv + g_push_constants.m_inv_src_resolution * float2(l_x, l_y);
                float3 l_val = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).rgb;

                // Get luminance from hdr color
                float l_cur_lum = get_luminance(l_val);

#if LOWER
                if (l_lum > l_cur_lum)
                {
                    l_lum = l_cur_lum;
                    l_color = float4(l_val, 1);
                }
#elif HIGHER
                if (l_lum < l_cur_lum)
                {
                    l_lum = l_cur_lum;
                    l_color = float4(l_val, 1);
                }
#endif

#endif // LIGHTING
            }
        }

    }
    else // Otherwise, downsample manually
    {
#if SINGLE_CHANNEL_UINT
        uint3 l_base_location = uint3(p_dispatch_thread_id.x << 1, p_dispatch_thread_id.y << 1, 0);
#elif SINGLE_CHANNEL_FLOAT || LIGHTING
        l_base_uv = l_base_uv + g_push_constants.m_inv_src_resolution * 0.5f;
#endif

        [unroll(9)]
        for (uint l_x = 0; l_x < l_step.x; l_x++)
        {
            for (uint l_y = 0; l_y < l_step.y; l_y++)
            {
#if SINGLE_CHANNEL_UINT
                uint3 l_location = l_base_location + uint3(l_x, l_y, 0);
                float l_val = l_texture.Load(l_location);

#if LOWER
                l_retval = min(l_retval, l_val);
#elif HIGHER
                l_retval = max(l_retval, l_val);
#endif

#elif SINGLE_CHANNEL_FLOAT
                float2 l_uv = l_base_uv + g_push_constants.m_inv_src_resolution * float2(l_x, l_y);
                float l_val = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).r;

#if LOWER
                l_color = min(l_color, l_val);
#elif HIGHER
                l_color = max(l_color, l_val);
#endif

#elif LIGHTING
                float2 l_uv = l_base_uv + g_push_constants.m_inv_src_resolution * float2(l_x, l_y);
                float3 l_val = bindless_tex2d_sample_level(g_push_constants.m_src_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv).rgb;

                // Get luminance from hdr color
                float l_cur_lum = get_luminance(l_val);

#if LOWER
                if (l_lum > l_cur_lum)
                {
                    l_lum = l_cur_lum;
                    l_color = float4(l_val, 1);
                }
#elif HIGHER
                if (l_lum < l_cur_lum)
                {
                    l_lum = l_cur_lum;
                    l_color = float4(l_val, 1);
                }
#endif

#endif // LIGHTING
            }
        }
    }

#if SINGLE_CHANNEL_UINT
    l_rt[p_dispatch_thread_id.xy] = l_retval;
#elif SINGLE_CHANNEL_FLOAT || LIGHTING
    l_rt[p_dispatch_thread_id.xy] = l_color;
#endif
}
