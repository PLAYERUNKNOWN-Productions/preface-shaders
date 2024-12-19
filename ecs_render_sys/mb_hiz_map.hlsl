// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// Push constants
ConstantBuffer<cb_push_hiz_map> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    Texture2D<float> l_src = ResourceDescriptorHeap[g_push_constants.m_src_srv];
    RWTexture2D<float> l_dst = ResourceDescriptorHeap[g_push_constants.m_dst_uav];
    
    uint2 l_dst_coords = uint2(p_dispatch_thread_id.x, p_dispatch_thread_id.y);
    uint2 l_src_coords = l_dst_coords * 2;

    uint2 l_src_dimensions;
    l_src.GetDimensions(l_src_dimensions.x, l_src_dimensions.y);

    uint2 l_dst_dimensions;
    l_dst.GetDimensions(l_dst_dimensions.x, l_dst_dimensions.y);

    float2 l_uv = (float2(l_src_coords) + .5f) / float2(l_src_dimensions);
    //float l_sample = l_src.Sample((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv);
    float4 l_samples = l_src.Gather((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv);
    float l_sample = min(min(l_samples.x, l_samples.y), min(l_samples.z, l_samples.w));

    //todo we could early out if l_sample every becomes 0

    if(l_src_dimensions.x & 1)//we are reducing an odd width texture
    {
        if(p_dispatch_thread_id.x == l_dst_dimensions.x - 1)
        {
            float2 l_uv_extra = l_uv + float2( rcp(l_src_dimensions.x), 0);
            //float l_sample_extra = l_src.Sample((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv_extra);
            float4 l_samples_extra = l_src.Gather((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv_extra);
            float l_sample_extra = min(min(l_samples_extra.x, l_samples_extra.y), min(l_samples_extra.z, l_samples_extra.w));

            l_sample = min(l_sample, l_sample_extra);
        }
    }

    if(l_src_dimensions.y & 1)//we are reducing an odd height texture
    {
        if(p_dispatch_thread_id.y == l_dst_dimensions.y - 1)
        {
            float2 l_uv_extra = l_uv + float2( 0, rcp(l_src_dimensions.y));
            //float l_sample_extra = l_src.Sample((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv_extra);
            float4 l_samples_extra = l_src.Gather((SamplerState)SamplerDescriptorHeap[SAMPLER_MIN_CLAMP], l_uv_extra);
            float l_sample_extra = min(min(l_samples_extra.x, l_samples_extra.y), min(l_samples_extra.z, l_samples_extra.w));

            l_sample = min(l_sample, l_sample_extra);
        }
    }

    if((l_src_dimensions.x & 1) && (l_src_dimensions.y & 1))
    {
        if((p_dispatch_thread_id.x == l_dst_dimensions.x - 1) && (p_dispatch_thread_id.y == l_dst_dimensions.y - 1))
        {
            //special case for the bottom right pixel when both texture dimensions are odd
            float l_sample_extra = l_src.Load(uint3(l_src_dimensions - 1, 0));

            l_sample = min(l_sample, l_sample_extra);
        }
    }

    l_dst[l_dst_coords] = l_sample;
}