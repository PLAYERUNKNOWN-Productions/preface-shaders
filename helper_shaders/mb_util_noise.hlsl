// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_NOISE_HLSL
#define MB_SHADER_NOISE_HLSL

// 1 / 289
#define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744

float mod289(float p_x)
{
    return p_x - floor(p_x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

float2 mod289(float2 p_x)
{
    return p_x - floor(p_x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

float3 mod289(float3 p_x)
{
    return p_x - floor(p_x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

float4 mod289(float4 p_x)
{
    return p_x - floor(p_x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

// ( x*34.0 + 1.0 )*x =
// x*x*34.0 + x
float permute(float p_x)
{
    return mod289(p_x*p_x*34.0 + p_x);
}

float3 permute(float3 p_x)
{
    return mod289(p_x*p_x*34.0 + p_x);
}

float4 permute(float4 p_x)
{
    return mod289(p_x*p_x*34.0 + p_x);
}

float4 grad4(float p_j, float4 p_ip)
{
    const float4 l_ones = float4(1.0, 1.0, 1.0, -1.0);
    float4 l_p, l_s;
    l_p.xyz = floor( frac(p_j * p_ip.xyz) * 7.0) * p_ip.z - 1.0;
    l_p.w = 1.5 - dot( abs(l_p.xyz), l_ones.xyz );

    // GLSL: lessThan(x, y) = x < y
    // HLSL: 1 - step(y, x) = x < y
    l_p.xyz -= sign(l_p.xyz) * (l_p.w < 0);

    return l_p;
}

float simplex_noise_2d(float2 p_v)
{
    const float4 l_c = float4(
        0.211324865405187,  // (3.0-sqrt(3.0))/6.0
        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
        -0.577350269189626, // -1.0 + 2.0 * l_c.x
        0.024390243902439   // 1.0 / 41.0
    );

    // First corner
    float2 l_i = floor( p_v + dot(p_v, l_c.yy) );
    float2 l_x0 = p_v - l_i + dot(l_i, l_c.xx);

    // Other corners
    // float2 l_i1 = (l_x0.x > l_x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    // Lex-DRL: afaik, step() in GPU is faster than if(), so:
    // step(x, y) = x <= y

    // Actually, a simple conditional without branching is faster than that madness :)
    int2 l_i1 = (l_x0.x > l_x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float4 l_x12 = l_x0.xyxy + l_c.xxzz;
    l_x12.xy -= l_i1;

    // Permutations
    l_i = mod289(l_i); // Avoid truncation effects in permutation
    float3 l_p = permute(
        permute(
                l_i.y + float3(0.0, l_i1.y, 1.0 )
        ) + l_i.x + float3(0.0, l_i1.x, 1.0 )
    );

    float3 l_m = max(
        0.5 - float3(
            dot(l_x0, l_x0),
            dot(l_x12.xy, l_x12.xy),
            dot(l_x12.zw, l_x12.zw)
        ),
        0.0
    );
    l_m = l_m*l_m ;
    l_m = l_m*l_m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    float3 l_x = 2.0 * frac(l_p * l_c.www) - 1.0;
    float3 l_h = abs(l_x) - 0.5;
    float3 l_ox = floor(l_x + 0.5);
    float3 l_a0 = l_x - l_ox;

    // Normalise gradients implicitly by scaling l_m
    // Approximation of: l_m *= inversesqrt( l_a0*l_a0 + l_h*l_h );
    l_m *= 1.79284291400159 - 0.85373472095314 * ( l_a0*l_a0 + l_h*l_h );

    // Compute final noise value at P
    float3 l_g;
    l_g.x = l_a0.x * l_x0.x + l_h.x * l_x0.y;
    l_g.yz = l_a0.yz * l_x12.xz + l_h.yz * l_x12.yw;
    return 130.0 * dot(l_m, l_g);
}

// Output [-1..1]
float simplex_noise_3d(float3 p_v)
{
    const float2 l_c = float2
    (
        0.166666666666666667, // 1/6
        0.333333333333333333  // 1/3
    );
    const float4 l_d = float4(0.0, 0.5, 1.0, 2.0);

    // First corner
    float3 l_i = floor( p_v + dot(p_v, l_c.yyy) );
    float3 l_x0 = p_v - l_i + dot(l_i, l_c.xxx);

    // Other corners
    float3 l_g = step(l_x0.yzx, l_x0.xyz);
    float3 l = 1 - l_g;
    float3 l_i1 = min(l_g.xyz, l.zxy);
    float3 l_i2 = max(l_g.xyz, l.zxy);

    float3 l_x1 = l_x0 - l_i1 + l_c.xxx;
    float3 l_x2 = l_x0 - l_i2 + l_c.yyy; // 2.0*l_c.x = 1/3 = l_c.y
    float3 l_x3 = l_x0 - l_d.yyy;      // -1.0+3.0*l_c.x = -0.5 = -l_d.y

    // Permutations
    l_i = mod289(l_i);
    float4 l_p = permute(
        permute(
            permute(
                    l_i.z + float4(0.0, l_i1.z, l_i2.z, 1.0 )
            ) + l_i.y + float4(0.0, l_i1.y, l_i2.y, 1.0 )
        )     + l_i.x + float4(0.0, l_i1.x, l_i2.x, 1.0 )
    );

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float l_n_ = 0.142857142857; // 1/7
    float3 l_ns = l_n_ * l_d.wyz - l_d.xzx;

    float4 p_j = l_p - 49.0 * floor(l_p * l_ns.z * l_ns.z); // mod(l_p,7*7)

    float4 l_x_ = floor(p_j * l_ns.z);
    float4 l_y_ = floor(p_j - 7.0 * l_x_ ); // mod(p_j,N)

    float4 l_x = l_x_ *l_ns.x + l_ns.yyyy;
    float4 l_y = l_y_ *l_ns.x + l_ns.yyyy;
    float4 l_h = 1.0 - abs(l_x) - abs(l_y);

    float4 l_b0 = float4( l_x.xy, l_y.xy );
    float4 l_b1 = float4( l_x.zw, l_y.zw );

    //float4 l_s0 = float4(lessThan(l_b0,0.0))*2.0 - 1.0;
    //float4 l_s1 = float4(lessThan(l_b1,0.0))*2.0 - 1.0;
    float4 l_s0 = floor(l_b0)*2.0 + 1.0;
    float4 l_s1 = floor(l_b1)*2.0 + 1.0;
    float4 l_sh = -step(l_h, 0.0);

    float4 l_a0 = l_b0.xzyw + l_s0.xzyw*l_sh.xxyy ;
    float4 l_a1 = l_b1.xzyw + l_s1.xzyw*l_sh.zzww ;

    float3 l_p0 = float3(l_a0.xy,l_h.x);
    float3 l_p1 = float3(l_a0.zw,l_h.y);
    float3 l_p2 = float3(l_a1.xy,l_h.z);
    float3 l_p3 = float3(l_a1.zw,l_h.w);

    //Normalise gradients
    float4 l_norm = rsqrt(float4(   dot(l_p0, l_p0),
                                    dot(l_p1, l_p1),
                                    dot(l_p2, l_p2),
                                    dot(l_p3, l_p3)));
    l_p0 *= l_norm.x;
    l_p1 *= l_norm.y;
    l_p2 *= l_norm.z;
    l_p3 *= l_norm.w;

    // Mix final noise value
    float4 l_m = max(
        0.6 - float4(
            dot(l_x0, l_x0),
            dot(l_x1, l_x1),
            dot(l_x2, l_x2),
            dot(l_x3, l_x3)
        ),
        0.0
    );
    l_m = l_m * l_m;
    return 42.0 * dot(
        l_m*l_m,
        float4(
            dot(l_p0, l_x0),
            dot(l_p1, l_x1),
            dot(l_p2, l_x2),
            dot(l_p3, l_x3)
        )
    );
}

float random_1d(float p_seed)
{
    return frac(sin(p_seed) * 43758.5453);
}

float random_2d(float2 p_seed)
{
    return frac(sin(dot(p_seed.xy, float2(12.9898f, 78.233f))) * 43758.5453123f);
}

float random_3d(float3 seed)
{
    return frac(sin(dot(seed, float3(12.9898f, 78.233f, 45.543f))) * 43758.5453123f);
}

float3 hash_3(float2 p_value)
{
    float3 l_randVals = frac(
        sin(float3(
            dot(p_value, float2(127.1, 311.7)),
            dot(p_value, float2(269.5, 183.3)),
            dot(p_value, float2(195.7, 225.4))
        )) * 43758.5453
    );
    return l_randVals;
}

float3 hash_3(float3 p_value)
{
    float3 l_randVals = frac(
        sin(float3(
            dot(p_value, float3(127.1, 311.7, 538.3)),
            dot(p_value, float3(269.5, 183.3, 245.2)),
            dot(p_value, float3(195.7, 225.4, 843.5))
        )) * 43758.5453
    );
    return l_randVals;
}

// https://www.shadertoy.com/view/XlXcW4
float3 hash_3(uint3 p_value)
{
    const uint k = 1103515245U;

    p_value = ((p_value >> 8U) ^ p_value.yzx) * k;
    p_value = ((p_value >> 8U) ^ p_value.yzx) * k;
    p_value = ((p_value >> 8U) ^ p_value.yzx) * k;

    return p_value * (1.0 / float(0xffffffffU));
}

float2 hash_2(int2 x)
{
    const int k = 1103515245;

    x = ((x>>8U)^x.yx)*k;
    x = ((x>>8U)^x.yx)*k;
    x = ((x>>8U)^x.yx)*k;

    return float2(x) / float(0xffffffffU);
}

float3 hash_3(int3 x)
{
    const int k = 1103515245;

    x = (( x >> 8U ) ^ x.yzx) * k;
    x = (( x >> 8U ) ^ x.yzx) * k;
    x = (( x >> 8U ) ^ x.yzx) * k;

    return x / float(0xffffffffU);
}

// Shader toy: https://www.shadertoy.com/view/4dS3Wd

float hash(float p_value)
{
    p_value = frac(p_value * 0.011);
    p_value *= p_value + 7.5;
    p_value *= p_value + p_value;
    return frac(p_value);
}

float hash(float2 p_value)
{
    float3 l_p3 = frac(float3(p_value.xyx) * 0.13);
    l_p3 += dot(l_p3, l_p3.yzx + 3.333);
    return frac((l_p3.x + l_p3.y) * l_p3.z);
}

float fbm_octave(float p_value)
{
    float l_i = floor(p_value);
    float l_f = frac(p_value);
    float l_u = l_f * l_f * (3.0 - 2.0 * l_f);
    return lerp(hash(l_i), hash(l_i + 1.0), l_u);
}

float fbm_octave(float2 l_seed)
{
    float2 l_floor = floor(l_seed);
    float2 l_frac = frac(l_seed);

    // Four corners in 2D of a tile
    float l_a = random_2d(l_floor);
    float l_b = random_2d(l_floor + float2(1.0, 0.0));
    float l_c = random_2d(l_floor + float2(0.0, 1.0));
    float l_d = random_2d(l_floor + float2(1.0, 1.0));

    float2 l_u = l_frac * l_frac * (3.0 - 2.0 * l_frac);

    return lerp(l_a, l_b, l_u.x) + (l_c - l_a) * l_u.y * (1.0 - l_u.x) + (l_d - l_b) * l_u.x * l_u.y;
}

float fbm_octave(float3 l_val)
{
    float3 l_a = floor(l_val);
    float3 l_d = l_val - l_a;
    l_d = l_d * l_d * (3.0 - 2.0 * l_d);

    float4 l_b = l_a.xxyy + float4(0.0, 1.0, 0.0, 1.0);
    float4 l_k1 = permute(l_b.xyxy);
    float4 l_k2 = permute(l_k1.xyxy + l_b.zzww);

    float4 l_c = l_k2 + l_a.zzzz;
    float4 l_k3 = permute(l_c);
    float4 l_k4 = permute(l_c + 1.0);

    float4 l_o1 = frac(l_k3 * (1.0 / 41.0));
    float4 l_o2 = frac(l_k4 * (1.0 / 41.0));

    float4 l_o3 = l_o2 * l_d.z + l_o1 * (1.0 - l_d.z);
    float2 l_o4 = l_o3.yw * l_d.x + l_o3.xz * (1.0 - l_d.x);

    return l_o4.y * l_d.y + l_o4.x * (1.0 - l_d.y);
}

float fbm_1d(float p_seed, int p_num_octaves, float p_base_amplitude, float p_amplitude_gain = 0.5f, float p_lacunarity = 2.0f, float p_lacunarity_shift = 100.0f)
{
    float l_v = 0.0;
    float l_amplitude = p_base_amplitude;
    for (int l_i = 0; l_i < p_num_octaves; ++l_i)
    {
        l_v += l_amplitude * fbm_octave(p_seed);
        p_seed = p_seed * p_lacunarity + p_lacunarity_shift;
        l_amplitude *= p_amplitude_gain;
    }
    return l_v;
}

float fbm_2d(float2 p_seed, int p_num_octaves, float p_base_amplitude, float p_amplitude_gain = 0.5f, float p_lacunarity = 2.0f, float p_lacunarity_shift = 100.0f)
{
    float l_v = 0.0;
    float l_amplitude = p_base_amplitude;

    // Rotate to reduce axial bias
    float2x2 rot = float2x2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
    for (int l_i = 0; l_i < p_num_octaves; ++l_i)
    {
        l_v += l_amplitude * fbm_octave(p_seed);
        p_seed = mul(rot, p_seed) * p_lacunarity + p_lacunarity_shift;
        l_amplitude *= p_amplitude_gain;
    }
    return l_v;
}

float fbm_3d(float3 p_seed, int p_num_octaves, float p_base_amplitude, float p_amplitude_gain = 0.5f, float p_lacunarity = 2.0f, float p_lacunarity_shift = 100.0f)
{
    float l_v = 0.0;
    float l_amplitude = p_base_amplitude;
    for (int l_i = 0; l_i < p_num_octaves; ++l_i)
    {
        l_v += l_amplitude * fbm_octave(p_seed);
        p_seed = p_seed * p_lacunarity + p_lacunarity_shift;
        l_amplitude *= p_amplitude_gain;
    }
    return l_v;
}


//////////////////////////////////////////////////////
// REPEATING NOISES USING FRACTURED CAMERA POSITION //
//////////////////////////////////////////////////////

// 2D // range: 0-1
float noise_repeat(float2 p_pos, int p_scale)
{
    int2 l_index = int2(floor(p_pos));
    float2 l_factor = frac(p_pos);

	float2 l_weight = l_factor*l_factor*(3.0-2.0*l_factor);

    return lerp(lerp(dot(hash_2((l_index + int2(0,0)) % p_scale), l_factor - float2(0.0,0.0)),
                     dot(hash_2((l_index + int2(1,0)) % p_scale), l_factor - float2(1.0,0.0)), l_weight.x),
                lerp(dot(hash_2((l_index + int2(0,1)) % p_scale), l_factor - float2(0.0,1.0)),
                     dot(hash_2((l_index + int2(1,1)) % p_scale), l_factor - float2(1.0,1.0)), l_weight.x), l_weight.y) + 0.5;
}

// 3D // range: 0-1
float noise_repeat(float3 p_pos, int p_scale)
{
    int3 l_index = int3(floor(p_pos));
    float3 l_factor = frac(p_pos);

	float3 l_weight = l_factor*l_factor*(3.0-2.0*l_factor);

    return lerp(lerp(lerp(dot(hash_3((l_index + int3(0,0,0)) % p_scale), l_factor - float3(0.0,0.0,0.0)),
                          dot(hash_3((l_index + int3(1,0,0)) % p_scale), l_factor - float3(1.0,0.0,0.0)), l_weight.x),
                     lerp(dot(hash_3((l_index + int3(0,1,0)) % p_scale), l_factor - float3(0.0,1.0,0.0)),
                          dot(hash_3((l_index + int3(1,1,0)) % p_scale), l_factor - float3(1.0,1.0,0.0)), l_weight.x), l_weight.y),
                lerp(lerp(dot(hash_3((l_index + int3(0,0,1)) % p_scale), l_factor - float3(0.0,0.0,1.0)),
                          dot(hash_3((l_index + int3(1,0,1)) % p_scale), l_factor - float3(1.0,0.0,1.0)), l_weight.x),
                     lerp(dot(hash_3((l_index + int3(0,1,1)) % p_scale), l_factor - float3(0.0,1.0,1.0)),
                          dot(hash_3((l_index + int3(1,1,1)) % p_scale), l_factor - float3(1.0,1.0,1.0)), l_weight.x), l_weight.y), l_weight.z) + 0.5;
}

// 2D // range: 0-1
float voronoi_repeat(in float2 uv, in int repeat)
{
    int2 uv_floor = int2(floor(uv));
    float2 uv_fract = frac(uv);
    float sum = 1000.0;

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            int2 index = int2(i,j);

            int2 fi = (uv_floor + index) % repeat;

            float dst = distance(uv_fract, hash(fi) + float2(index));
            sum = min(dst, sum);
        }
    }
    return sin(acos(saturate(sum * 0.9))); // Multiply by 0.9 because distance can be more than 1 breaking the acos()
}

float fbm_repeat(float2 p_uv, int p_scale, int p_steps)
{
    float noise = 0.0;
    for(int i = 0; i < p_steps; i++)
    {
        int l_scale_sqr = 2U << i;
        int l_final_scale = p_scale * l_scale_sqr;

        float l_fbm_noise = voronoi_repeat(p_uv * l_final_scale, l_final_scale);

        noise += l_fbm_noise / l_scale_sqr;
    }
    float fraction = 1.0 - pow(2.0, -p_steps);
    return noise / fraction;
}

// 2D //
float noise_1(float2 p_pos, cb_camera_t p_cam, int p_scale) // 1 meter
{
    p_pos = frac(p_pos / 1.0 + p_cam.m_camera_pos_frac_1.xy) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

float noise_100(float2 p_pos, cb_camera_t p_cam, int p_scale) // 100 meter
{
    p_pos = frac(p_pos / 100.0 + p_cam.m_camera_pos_frac_100.xy) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

float noise_10000(float2 p_pos, cb_camera_t p_cam, int p_scale) // 10000 meter
{
    p_pos = frac(p_pos / 10000.0 + p_cam.m_camera_pos_frac_10000.xy) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

// 3D //
float noise_1(float3 p_pos, cb_camera_t p_cam, int p_scale) // 1 meter
{
    p_pos = frac(p_pos / 1.0 + p_cam.m_camera_pos_frac_1) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

float noise_100(float3 p_pos, cb_camera_t p_cam, int p_scale) // 100 meter
{
    p_pos = frac(p_pos / 100.0 + p_cam.m_camera_pos_frac_100) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

float noise_10000(float3 p_pos, cb_camera_t p_cam, int p_scale) // 10000 meter
{
    p_pos = frac(p_pos / 10000.0 + p_cam.m_camera_pos_frac_10000) * p_scale;
    return noise_repeat(p_pos, p_scale);
}

float voronoi(float3 pos)
{
    float3 tile_index = floor(pos);
    float3 tile_gradient = frac(pos);
    float sum = 1000.0;

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            for(int k = -1; k <= 1; k++)
            {
                float3 index = float3(i,j,k);
                float3 localPos = tile_gradient - (hash_3(tile_index + index) + index);
                float dst = dot(localPos, localPos);
                sum = min(dst, sum);
            }
        }
    }
    return sqrt(sum);
}

float voronoi(float2 pos)
{
    float2 tile_index = floor(pos);
    float2 tile_gradient = frac(pos);
    float sum = 1000.0;

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            float2 index = float2(i,j);
            float2 localPos = tile_gradient - (hash_3(float3(tile_index + index, 0.0)).xy + index);
            float dst = dot(localPos, localPos);
            sum = min(dst, sum);
        }
    }
    return sqrt(sum);
}

float voronoi_fbm(float3 pos, int itt, float p_amplitude_gain = 0.5f, float p_lacunarity = 2.0f)
{
    float uv_scale_itt = p_lacunarity;
    float sum_intensity_itt = p_amplitude_gain;

    float sum = 0.0;
    float noise_intensity = 1.0;

    for(int i = 0; i < itt; i++)
    {
        float noise = voronoi(pos);
        noise *= noise_intensity;

        sum += noise;

        pos *= uv_scale_itt;
        noise_intensity *= sum_intensity_itt;
    }

    // Calculate the maximum possible sum
    float max_sum = (1.0 - pow(sum_intensity_itt, itt)) / (1.0 - sum_intensity_itt);

    // Normalize to 0-1 range
    return sum / max_sum;
}

float voronoi_fbm(float2 pos, int itt, float p_amplitude_gain = 0.5f, float p_lacunarity = 2.0f)
{
    float uv_scale_itt = p_lacunarity;
    float sum_intensity_itt = p_amplitude_gain;

    float sum = 0.0;
    float noise_intensity = 1.0;

    for(int i = 0; i < itt; i++)
    {
        float noise = voronoi(pos);
        noise *= noise_intensity;

        sum += noise;

        pos *= uv_scale_itt;
        noise_intensity *= sum_intensity_itt;
    }

    // Calculate the maximum possible sum
    float max_sum = (1.0 - pow(sum_intensity_itt, itt)) / (1.0 - sum_intensity_itt);

    // Normalize to 0-1 range
    return sum / max_sum;
}

#endif // MB_SHADER_NOISE_HLSL
