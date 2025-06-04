// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_VERTEX_UTILITIES_HLSL
#define MB_SHADER_VERTEX_UTILITIES_HLSL

// Helper functions
#include "../shared_shaders/mb_shared_buffers.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"

float3 rotate_y(float3 vec, float theta)
{
    float c = cos(theta);
    float s = sin(theta);
    float3x3 mat = float3x3(c, 0, s,
        0, 1, 0,
        -s, 0, c);

    return mul(vec, mat);
}

// Rotation matrix around axis
float3 rotate_axis(float3 input, float angle, float3 axis)
{
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;

    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    float3x3 mat = float3x3(t * x * x + c, t * x * y - s * z, t * x * z + s * y,
        t * x * y + s * z, t * y * y + c, t * y * z - s * x,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c);

    return mul(input, mat);
}

//  Wind is applied as sin(a+b), where a is a large value, and b is a small value. If we add them
// and compute the sin, we lose a lot of precision before calculating the sine. By splitting the
// sin(a+b) into it's sum identity, we do 3x more trigonometry, but keep the precision
float3 sin_sum(float3 a, float b)
{
    return sin(a)*cos(b) + cos(a)*sin(b);
}

float3 apply_wind_to_position(float3 p_position, float4x3 p_transform, uint p_time, cb_camera_t p_cam, bool p_wind_small)
{
    float l_time_sec = p_time * 0.001;

    float3 l_camera_local_pos_pivot = p_transform._41_42_43;
    float3 l_pivot_pos_ws = l_camera_local_pos_pivot + p_cam.m_camera_pos;
    float3 l_rand = hash_3(float3(p_transform._11, p_transform._22, p_transform._33)); // random hash value based on transformation matrix

    float3 l_planet_normal = normalize(l_pivot_pos_ws);
    float3 l_planet_tangent = cross(l_planet_normal, float3(0.0, 1.0, 0.0));
    // First transform wind from world dir to local dir. After add world dir for random directions
    float3 l_wind_dir = normalize(mul(l_planet_tangent, transpose((float3x3)p_transform)) + l_planet_tangent * l_rand * 2.0);

    float3 l_tree_noise_3 = sin_sum(l_pivot_pos_ws * 0.04, l_time_sec * 0.4);
    float l_tree_noise = l_tree_noise_3.x + l_tree_noise_3.y + l_tree_noise_3.z;

    p_position = rotate_axis(p_position, l_tree_noise * 0.0004 * p_position.y, l_wind_dir);

    if(p_wind_small)
    {
        float l_veg_mov_factor = dot(rotate_y(l_wind_dir, M_PI * 0.5).xz, normalize(p_position.xz));
        float l_veg_mov_noise = sin(l_rand.x + l_rand.y + l_rand.z + p_position.y - l_time_sec * 1.5);
        p_position = rotate_y(p_position, l_veg_mov_factor * l_veg_mov_noise * 0.025);
    }

    return p_position;
}

#endif // MB_SHADER_VERTEX_UTILITIES_HLSL
