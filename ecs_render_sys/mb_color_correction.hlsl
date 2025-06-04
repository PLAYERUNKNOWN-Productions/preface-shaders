// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Apply color correction by using a LUT table
float3 apply_color_correction(float3 input_color, uint lut_srv, uint lut_size) : SV_TARGET
{
    if (lut_srv == RAL_NULL_BINDLESS_INDEX)
    {
        return input_color;
    }

    // https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-24-using-lookup-tables-accelerate-color
    float3 input_uv = input_color * (lut_size - 1.0)/lut_size + 1.f / (2.f * lut_size);

    Texture3D texture = ResourceDescriptorHeap[lut_srv];
    float3 remapped_color = texture.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], input_uv, 0).rgb;
    return remapped_color;
}
