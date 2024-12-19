// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
float3 PositivePow(float3 base, float3 power)
{
    return pow(abs(base), power);
}

//-----------------------------------------------------------------------------
// sRGB
float3 SRGBToLinear(float3 c)
{
    float3 linearRGBLo = c / 12.92;
    float3 linearRGBHi = PositivePow((c + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
    float3 linearRGB = (c <= 0.04045) ? linearRGBLo : linearRGBHi;
    return linearRGB;
}

//-----------------------------------------------------------------------------
// sRGB
float3 LinearToSRGB(float3 c)
{
    float3 sRGBLo = c * 12.92;
    float3 sRGBHi = (PositivePow(c, float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) * 1.055) - 0.055;
    float3 sRGB = (c <= 0.0031308) ? sRGBLo : sRGBHi;
    return sRGB;
}

//-----------------------------------------------------------------------------
void LoadTexelsRG(Texture2D tex,
                  SamplerState samp,
                  float2 inv_texture_width,
                  uint2 dispatch_thread_id,
                  out float block_r[16],
                  out float block_g[16])
{
    // Gather from bottom-right corner
    float2 uv = (dispatch_thread_id * 4 + 1.0) * inv_texture_width;

    // We multiple R and G channel to support both RG and AG formats
    float4 r = tex.GatherRed(samp, uv, int2(0, 0));
    float4 g = tex.GatherGreen(samp, uv, int2(0, 0));
    float4 a = tex.GatherAlpha(samp, uv, int2(0, 0));
    block_r[0] = r[3] * a[3]; block_g[0] = g[3];
    block_r[1] = r[2] * a[2]; block_g[1] = g[2];
    block_r[4] = r[0] * a[0]; block_g[4] = g[0];
    block_r[5] = r[1] * a[1]; block_g[5] = g[1];

    r = tex.GatherRed(samp, uv, int2(2, 0));
    g = tex.GatherGreen(samp, uv, int2(2, 0));
    a = tex.GatherAlpha(samp, uv, int2(2, 0));
    block_r[2] = r[3] * a[3]; block_g[2] = g[3];
    block_r[3] = r[2] * a[2]; block_g[3] = g[2];
    block_r[6] = r[0] * a[0]; block_g[6] = g[0];
    block_r[7] = r[1] * a[1]; block_g[7] = g[1];

    r = tex.GatherRed(samp, uv, int2(0, 2));
    g = tex.GatherGreen(samp, uv, int2(0, 2));
    a = tex.GatherAlpha(samp, uv, int2(0, 2));
    block_r[8] = r[3] * a[3]; block_g[8] = g[3];
    block_r[9] = r[2] * a[2]; block_g[9] = g[2];
    block_r[12] = r[0] * a[0]; block_g[12] = g[0];
    block_r[13] = r[1] * a[1]; block_g[13] = g[1];

    r = tex.GatherRed(samp, uv, int2(2, 2));
    g = tex.GatherGreen(samp, uv, int2(2, 2));
    a = tex.GatherAlpha(samp, uv, int2(2, 2));
    block_r[10] = r[3] * a[3]; block_g[10] = g[3];
    block_r[11] = r[2] * a[2]; block_g[11] = g[2];
    block_r[14] = r[0] * a[0]; block_g[14] = g[0];
    block_r[15] = r[1] * a[1]; block_g[15] = g[1];
}

//-----------------------------------------------------------------------------
void LoadTexelsRGB(Texture2D tex,
                   SamplerState samp,
                   float2 inv_texture_width,
                   uint2 dispatch_thread_id,
                   float input_is_srgb,
                   out float3 block[16])
{
    // Gather from bottom-right corner
    float2 uv = (dispatch_thread_id * 4 + 1.0) * inv_texture_width;

    float4 r = tex.GatherRed(samp, uv, int2(0, 0));
    float4 g = tex.GatherGreen(samp, uv, int2(0, 0));
    float4 b = tex.GatherBlue(samp, uv, int2(0, 0));
    block[0] = float3(r[3], g[3], b[3]);
    block[1] = float3(r[2], g[2], b[2]);
    block[4] = float3(r[0], g[0], b[0]);
    block[5] = float3(r[1], g[1], b[1]);

    r = tex.GatherRed(samp, uv, int2(2, 0));
    g = tex.GatherGreen(samp, uv, int2(2, 0));
    b = tex.GatherBlue(samp, uv, int2(2, 0));
    block[2] = float3(r[3], g[3], b[3]);
    block[3] = float3(r[2], g[2], b[2]);
    block[6] = float3(r[0], g[0], b[0]);
    block[7] = float3(r[1], g[1], b[1]);

    r = tex.GatherRed(samp, uv, int2(0, 2));
    g = tex.GatherGreen(samp, uv, int2(0, 2));
    b = tex.GatherBlue(samp, uv, int2(0, 2));
    block[8] = float3(r[3], g[3], b[3]);
    block[9] = float3(r[2], g[2], b[2]);
    block[12] = float3(r[0], g[0], b[0]);
    block[13] = float3(r[1], g[1], b[1]);

    r = tex.GatherRed(samp, uv, int2(2, 2));
    g = tex.GatherGreen(samp, uv, int2(2, 2));
    b = tex.GatherBlue(samp, uv, int2(2, 2));
    block[10] = float3(r[3], g[3], b[3]);
    block[11] = float3(r[2], g[2], b[2]);
    block[14] = float3(r[0], g[0], b[0]);
    block[15] = float3(r[1], g[1], b[1]);

    // TODO check asm
    [branch]
    if(input_is_srgb > 0)
    {
        [unroll]
        for (int i = 0; i < 16; ++i)
        {
            block[i] = LinearToSRGB(block[i]);
        }
    }
}

//-----------------------------------------------------------------------------
void LoadTexelsRGBA(Texture2D tex,
                    SamplerState samp,
                    uint2 dispatch_thread_id,
                    float input_is_srgb,
                    out float3 blockRGB[16],
                    out float blockA[16])
{
    float4 color;
    int3 pixel_coord = int3(dispatch_thread_id * 4, 0);

    // TODO check asm
    [branch]
    if (input_is_srgb > 0)
    {
        [unroll]
        for(int x = 0; x < 4; ++x)
        {
            [unroll]
            for(int y = 0; y < 4; ++y)
            {
                color = tex.Load(pixel_coord, int2(x, y));
                color.rgb = LinearToSRGB(color.rgb);

                int index = 4 * y + x;
                blockRGB[index] = color.rgb;
                blockA[index] = color.a;
            }
        }
    }else
    {
        [unroll]
        for (int x = 0; x < 4; ++x)
        {
            [unroll]
            for (int y = 0; y < 4; ++y)
            {
                color = tex.Load(pixel_coord, int2(x, y));

                int index = 4 * y + x;
                blockRGB[index] = color.rgb;
                blockA[index] = color.a;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// TODO: research a better endpoint search
void GetMinMaxRG(float block_r[16],
                 float block_g[16],
                 out float min_color_r,
                 out float max_color_r,
                 out float min_color_g,
                 out float max_color_g)
{
    min_color_r = block_r[0];
    max_color_r = block_r[0];
    min_color_g = block_g[0];
    max_color_g = block_g[0];

    [unroll]
    for (int i = 1; i < 16; ++i)
    {
        min_color_r = min(min_color_r, block_r[i]);
        max_color_r = max(max_color_r, block_r[i]);
        min_color_g = min(min_color_g, block_g[i]);
        max_color_g = max(max_color_g, block_g[i]);
    }
}

//-----------------------------------------------------------------------------
void GetMinMaxRGB(float3 block[16], out float3 min_color, out float3 max_color)
{
    min_color = block[0];
    max_color = block[0];

    for (int i = 1; i < 16; ++i)
    {
        min_color = min(min_color, block[i]);
        max_color = max(max_color, block[i]);
    }
}

//-----------------------------------------------------------------------------
void GetMinMaxChannel(float block[16], out float min_channel, out float max_channel)
{
    min_channel = block[0];
    max_channel = block[0];

    for (int i = 1; i < 16; ++i)
    {
        min_channel = min(min_channel, block[i]);
        max_channel = max(max_channel, block[i]);
    }
}

//-----------------------------------------------------------------------------
void InsetMinMaxRGB(inout float3 min_color, inout float3 max_color)
{
    // Since we have four points, (1/16) * (max-min) will give us half the distance between
    //  two points on the line in color space
    float3 offset = (1.0f / 16.0f) * (max_color - min_color);

    // After applying the offset, we want to round up or down to the next integral color value (0 to 255)
    max_color = ceil((max_color - offset) * 255.0f) / 255.0f;
    min_color = floor((min_color + offset) * 255.0f) / 255.0f;
}

//-----------------------------------------------------------------------------
// TODO: try using dot's instead?
// Compres to 5:6:5 format
uint ColorTo565(float3 color)
{
    uint3 rgb = round(color * float3(31.0f, 63.0f, 31.0f));
    return (rgb.r << 11) | (rgb.g << 5) | rgb.b;
}

//-----------------------------------------------------------------------------
// Original BCn compression function
uint GetIndicesRGB(float3 block[16], float3 min_color, float3 max_color)
{
    uint indices = 0;

    // For each input color, we need to select between one of the following output colors:
    //  0: max_color
    //  1: (2/3)*max_color + (1/3)*min_color
    //  2: (1/3)*max_color + (2/3)*min_color
    //  3: min_color  
    //
    // We essentially just project (block[i] - max_color) onto (min_color - max_color), but we pull out
    //  a few constant terms.
    float3 diag = min_color - max_color;
    float step_inc = 3.0f / dot(diag, diag); // Scale up by 3, because our indices are between 0 and 3
    diag *= step_inc;
    float c = step_inc * (dot(max_color, max_color) - dot(max_color, min_color));

    for (int i = 15; i >= 0; --i)
    {
        // Compute the index for this block element
        uint index = round(dot(block[i], diag) + c);

        // Now we need to convert our index into the somewhat unintuivive BC1 indexing scheme:
        //  0: max_color
        //  1: min_color
        //  2: (2/3)*max_color + (1/3)*min_color
        //  3: (1/3)*max_color + (2/3)*min_color
        //
        // The mapping is:
        //  0 -> 0
        //  1 -> 2
        //  2 -> 3
        //  3 -> 1
        //
        // We can perform this mapping using bitwise operations, which is faster
        //  than predication or branching as long as it doesn't increase our register
        //  count too much. The mapping in binary looks like:
        //  00 -> 00
        //  01 -> 10
        //  10 -> 11
        //  11 -> 01
        //
        // Splitting it up by bit, the output looks like:
        //  bit1_out = bit0_in XOR bit1_in
        //  bit0_out = bit1_in 
        uint bit0_in = index & 1;
        uint bit1_in = index >> 1;
        indices |= ((bit0_in ^ bit1_in) << 1) | bit1_in;

        if (i != 0)
        {
            indices <<= 2;
        }
    }

    return indices;
}

//-----------------------------------------------------------------------------
// Original BCn compression function
void GetIndicesAlpha(float block[16], float min_alpha, float max_alpha, inout uint2 packed)
{
    float d = min_alpha - max_alpha;
    float step_inc = 7.0f / d;

    // Both packed.x and packed.y contain index values, so we need two loops

    uint index = 0;
    uint shift = 16;
    for (int i = 0; i < 6; ++i)
    {
        // For each input alpha value, we need to select between one of eight output values
        //  0: max_alpha
        //  1: (6/7)*max_alpha + (1/7)*min_alpha
        //  ...
        //  6: (1/7)*max_alpha + (6/3)*min_alpha
        //  7: min_alpha  
        index = round(step_inc * (block[i] - max_alpha));

        // Now we need to convert our index into the BC indexing scheme:
        //  0: max_alpha
        //  1: min_alpha
        //  2: (6/7)*max_alpha + (1/7)*min_alpha
        //  ...
        //  7: (1/7)*max_alpha + (6/3)*min_alpha
        index += (index > 0) - (7 * (index == 7));

        packed.x |= (index << shift);
        shift += 3;
    }

    // The 6th index straddles the two uints
    packed.y |= (index >> 1);

    shift = 2;
    for (i = 6; i < 16; ++i)
    {
        index = round((block[i] - max_alpha) * step_inc);
        index += (index > 0) - (7 * (index == 7));

        packed.y |= (index << shift);
        shift += 3;
    }
}
