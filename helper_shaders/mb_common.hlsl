// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_COMMON_HLSL
#define MB_SHADER_COMMON_HLSL

#include "../shared_shaders/mb_shared_common.hlsl"
#include "../shared_shaders/mb_shared_buffers.hlsl"

#define M_PI (3.14159265f)
#define M_INV_PI (1.0f / M_PI)
#define GAMMA (2.2f)
#define INV_GAMMA (1.0f / GAMMA)
#define FLT_MAX (3.402823466e+38f)

#define MB_CLIP_RANGE_END 500.f
#define MB_CLIP_GRADIENT_MAX 0.075f

// Debug
#define DEBUG_DIFFUSE               (0)
#define DEBUG_NORMAL                (0)
#define DEBUG_SPECULAR              (0)
#define DEBUG_ROUGHNESS             (0)
#define DEBUG_METALLIC              (0)
#define DEBUG_OCCLUSION             (0)
#define DEBUG_AOI                   (1)

#define ENABLE_IBL                  (1)
#define ENABLE_DIRECTIONAL_LIGHT    (1)
#define ENABLE_OMNI_LIGHT           (1)
#define ENABLE_SPOT_LIGHT           (1)

// This is a fake like term to take into account spherical shape of the earth
// Direct lighting part must be handled with shadows
// IBL part must be handled with probe grid that covers whole earth surface
#define ENABLE_FAKE_EARTH_SHADOW_TERM       (1)
#define ENABLE_FAKE_EARTH_SHADOW_TERM_IBL   (1)

// Z-buffer
// Inverse-Z is used to improve precision, so z-values for near and far plane are flipped
#define Z_NEAR 1.0f
#define Z_FAR 0.0f

// Macro pre-processing
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define _CONCAT(x,y) x##y
#define CONCAT(x,y) _CONCAT(x,y)

// User register define
#define USER_REGISTER(x, y) x, CONCAT(space,y)

// Push constants
#define REGISTER_PUSH_CONSTANTS USER_REGISTER(b0, MB_PUSH_CONSTANTS_SPACE)

// No resource bound
#define RAL_NULL_BINDLESS_INDEX 0xFFFFFFFF

#define V_BUFFERING_FRONT_FACE_BIT_MASK     0x80000000u
#define V_BUFFERING_WIND_BIT_MASK           0x40000000u
#define V_BUFFERING_WIND_SMALL_BIT_MASK     0x20000000u

uint pack_instance_id_pixel_options(uint p_instance_id, bool p_front_face, bool p_wind, bool p_wind_small)
{
    uint l_pixel_option_mask = V_BUFFERING_FRONT_FACE_BIT_MASK | V_BUFFERING_WIND_BIT_MASK | V_BUFFERING_WIND_SMALL_BIT_MASK;

    uint l_front_face_part  = p_front_face ? V_BUFFERING_FRONT_FACE_BIT_MASK : 0u;
    uint l_wind_part        = p_wind ? V_BUFFERING_WIND_BIT_MASK : 0u;
    uint l_wind_small_part  = p_wind_small ? V_BUFFERING_WIND_SMALL_BIT_MASK : 0u;

    uint l_instance_id_part = p_instance_id & ~l_pixel_option_mask;
    return l_front_face_part | l_wind_part | l_wind_small_part | l_instance_id_part;
}

void unpack_instance_id_pixel_options(uint p_packed, out uint p_instance_id, out bool p_front_face, out bool p_wind, out bool p_wind_small)
{
    uint l_pixel_option_mask = V_BUFFERING_FRONT_FACE_BIT_MASK | V_BUFFERING_WIND_BIT_MASK | V_BUFFERING_WIND_SMALL_BIT_MASK;
    p_instance_id = p_packed & ~l_pixel_option_mask;

    p_front_face    = (p_packed & V_BUFFERING_FRONT_FACE_BIT_MASK)  == V_BUFFERING_FRONT_FACE_BIT_MASK;
    p_wind          = (p_packed & V_BUFFERING_WIND_BIT_MASK)        == V_BUFFERING_WIND_BIT_MASK;
    p_wind_small    = (p_packed & V_BUFFERING_WIND_SMALL_BIT_MASK)  == V_BUFFERING_WIND_SMALL_BIT_MASK;
}

float4 bindless_tex2d_sample(   uint p_index,
                                SamplerState p_sampler,
                                float2 p_uv,
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                float2 p_uv_ddx,
                                float2 p_uv_ddy,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture2D l_texture = ResourceDescriptorHeap[(p_index)];
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        l_color = l_texture.SampleGrad(p_sampler, p_uv, p_uv_ddx, p_uv_ddy);
#else
        l_color = l_texture.Sample(p_sampler, p_uv);
#endif
    }
    return l_color;
}

float4 bindless_tex2d_sample_level( uint p_index,
                                    SamplerState p_sampler,
                                    float2 p_uv,
                                    float p_lod = 0,
                                    float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture2D l_texture = ResourceDescriptorHeap[p_index];
        l_color = l_texture.SampleLevel(p_sampler, p_uv, p_lod);
    }
    return l_color;
}

float4 bindless_tex2d_sample_bias(uint p_index,
                                  SamplerState p_sampler,
                                  float2 p_uv,
                                  float p_bias,
                                  float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture2D l_texture = ResourceDescriptorHeap[p_index];
        l_color = l_texture.SampleBias(p_sampler, p_uv, p_bias);
    }
    return l_color;
}

float4 bindless_tex2d_array_sample_level(   uint p_index,
                                            SamplerState p_sampler,
                                            float3 p_uv,
                                            float p_lod = 0,
                                            float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture2DArray l_texture = ResourceDescriptorHeap[NonUniformResourceIndex(p_index)];
        l_color = l_texture.SampleLevel(p_sampler, p_uv, p_lod);
    }
    return l_color;
}

float4 bindless_texcube_sample_level(uint p_index,
                                     SamplerState p_sampler,
                                     float3 p_uvw,
                                     float p_lod = 0,
                                     float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        TextureCube l_texture = ResourceDescriptorHeap[NonUniformResourceIndex(p_index)];
        l_color = l_texture.SampleLevel(p_sampler, p_uvw, p_lod);
    }
    return l_color;
}

float4 raytracing_bindless_tex2d_sample_level_with_feedback(uint p_index,
                                                            uint p_residency_buffer_srv,
                                                            uint2 p_residency_buffer_dim,
                                                            SamplerState p_sampler,
                                                            float2 p_uv,
                                                            float p_lod,
                                                            float4 p_default_val = 0)
{
    // Early exit if texture is not available
    if (p_index == RAL_NULL_BINDLESS_INDEX)
    {
        return p_default_val;
    }

    // Resolve resources
    Texture2D l_texture = ResourceDescriptorHeap[p_index];
    Buffer<uint> l_residency_map = ResourceDescriptorHeap[p_residency_buffer_srv];

    // Get index into the residency buffer
    int2 l_uv_int = frac(p_uv) * p_residency_buffer_dim;
    uint l_residency_buffer_index = l_uv_int.y * p_residency_buffer_dim.x + l_uv_int.x;
    uint l_min_mip_level = l_residency_map.Load(l_residency_buffer_index);

    //! /todo Do I need to handle this case manually? Check when vt is implemented
    // 255(0xFF) is used to indicate that tile is not covered by any loaded mip-level
    if(l_min_mip_level == 255)
    {
        return p_default_val;
    }

    // Clamped requested mip to min mip available
    l_min_mip_level = max(l_min_mip_level, p_lod);

    // Sample texture is passed as LOD parameter
    float4 l_color = l_texture.SampleLevel(p_sampler, p_uv, l_min_mip_level);
    return l_color;
}

float4 bindless_tex2d_sample_with_feedback( uint p_index,
                                            uint p_residency_buffer_srv,
                                            uint p_sampler_feedback_uav,
                                            uint2 p_residency_buffer_dim,
                                            SamplerState p_sampler,
                                            float2 p_uv,
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                            float2 p_uv_ddx,
                                            float2 p_uv_ddy,
#else
                                            float p_mip_lod_bias,
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
                                            float4 p_default_val = 0,
                                            uint2 p_screen_xy = 0)
{
    // Early exit if texture is not available
    if (p_index == RAL_NULL_BINDLESS_INDEX)
    {
        return p_default_val;
    }

    // Resolve resources
    Texture2D l_texture = ResourceDescriptorHeap[p_index];
    Buffer<uint> l_residency_map = ResourceDescriptorHeap[p_residency_buffer_srv];
    FeedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED> l_feedback_texture = ResourceDescriptorHeap[p_sampler_feedback_uav];

    // Get index into the residency buffer
    int2 l_uv_int = frac(p_uv) * p_residency_buffer_dim;
    uint l_residency_buffer_index = l_uv_int.y * p_residency_buffer_dim.x + l_uv_int.x;
    uint l_min_mip_level = l_residency_map.Load(l_residency_buffer_index);

    // Write sampler feedback before a possible early exit
    // Some vendors recommend doing it stochastically(AMD, NVIDIA)
#if defined(MB_VIRTUAL_TEXTURE_WRITE_FEEDBACK)
    // Improvement: we can make this animated, but so far no artifacts are seen from using just a static pattern
    // 4x4 seems to be granular enough
    if( p_screen_xy.x % 4 == 0 &&
        p_screen_xy.y % 4 == 0)
    {
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
        l_feedback_texture.WriteSamplerFeedbackGrad(l_texture, p_sampler, p_uv, p_uv_ddx, p_uv_ddy);
#else
        l_feedback_texture.WriteSamplerFeedbackBias(l_texture, p_sampler, p_uv, p_mip_lod_bias);
#endif
    }
#endif

    //! /todo Do I need to handle this case manually? Check when vt is implemented
    // 255(0xFF) is used to indicate that tile is not covered by any loaded mip-level
    if(l_min_mip_level == 255)
    {
        return p_default_val;
    }

    // Sample texture
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    float4 l_color = l_texture.SampleGrad(p_sampler, p_uv, p_uv_ddx, p_uv_ddy, 0, l_min_mip_level);
#else
    float4 l_color = l_texture.Sample(p_sampler, p_uv, 0, p_mip_lod_bias, l_min_mip_level);
#endif
    return l_color;
}

float4 bindless_tex2d_load( uint p_index,
                            uint3 p_location,
                            float4 p_default_val = 0)
{
    float4 l_color = p_default_val;
    if (p_index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture2D l_texture = ResourceDescriptorHeap[NonUniformResourceIndex(p_index)];
        l_color = l_texture.Load(p_location);
    }
    return l_color;
}

float4 bindless_tex3d_sample(   uint index,
                                SamplerState texture_sampler,
                                float3 uvw,
                                float4 default_val = 0)
{
    float4 color = default_val;
    if (index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture3D texture = ResourceDescriptorHeap[index];
        color = texture.Sample(texture_sampler, uvw);
    }
    return color;
}

float4 bindless_tex3d_sample_level( uint index,
                                    SamplerState texture_sampler,
                                    float3 uvw,
                                    float lod,
                                    float4 default_val = 0)
{
    float4 color = default_val;
    if (index != RAL_NULL_BINDLESS_INDEX)
    {
        Texture3D texture = ResourceDescriptorHeap[index];
        color = texture.SampleLevel(texture_sampler, uvw, lod);
    }
    return color;
}

float gamma_to_linear(float p_color)
{
    return pow(p_color, GAMMA);
}

float linear_to_gamma(float p_color)
{
    return pow(p_color, INV_GAMMA);
}

float2 gamma_to_linear(float2 p_color)
{
    return pow(p_color, GAMMA);
}

float2 linear_to_gamma(float2 p_color)
{
    return pow(p_color, INV_GAMMA);
}

float3 gamma_to_linear(float3 p_color)
{
    return pow(p_color, GAMMA);
}

float3 linear_to_gamma(float3 p_color)
{
    return pow(p_color, INV_GAMMA);
}

float get_luminance(float3 p_color)
{
#if 0 // Relative luminance: https://en.wikipedia.org/wiki/Relative_luminance
    return dot(p_color, float3(0.2126f, 0.7152f, 0.0722f));
#elif 1 // Perceived brightness : https://www.w3.org/TR/AERT/#color-contrast
    return dot(p_color, float3(0.299f, 0.587f, 0.114f));
#elif 0 // HSP Color Model: http://alienryderflex.com/hsp.html
    return sqrt(0.299f * p_color.x * p_color.x + 0.587 * p_color.y * p_color.y + 0.114 * p_color.z * p_color.z);
#endif
}

float3 rgb_to_hsl(float3 p_color)
{
    float4 l_k = float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    float4 l_p = lerp(float4(p_color.bg, l_k.wz), float4(p_color.gb, l_k.xy), step(p_color.b, p_color.g));
    float4 l_q = lerp(float4(l_p.xyw, p_color.r), float4(p_color.r, l_p.yzx), step(l_p.x, p_color.r));

    float l_d = l_q.x - min(l_q.w, l_q.y);
    float l_e = 1.0e-10;
    return float3(abs(l_q.z + (l_q.w - l_q.y) / (6.0 * l_d + l_e)), l_d / (l_q.x + l_e), l_q.x);
}

float3 hsl_to_rgb(float3 p_color)
{
    float4 l_k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 l_p = abs(frac(p_color.xxx + l_k.xyz) * 6.0 - l_k.www);
    return abs(p_color.z * lerp(l_k.xxx, clamp(l_p - l_k.xxx, 0.0, 1.0), p_color.y));
}

sb_geometry_pbr_material_t get_default_geometry_pbr_material()
{
    sb_geometry_pbr_material_t l_material;
    l_material.m_base_color_factor = 1.0f;
    l_material.m_base_color_texture_srv = RAL_NULL_BINDLESS_INDEX;
    l_material.m_roughness_factor = 1.0f;
    l_material.m_emissive_factor = 0;
    l_material.m_metallic_factor = 1.0f;
    l_material.m_metallic_roughness_texture_srv = RAL_NULL_BINDLESS_INDEX;
    l_material.m_normal_map_texture_srv = RAL_NULL_BINDLESS_INDEX;
    l_material.m_occlusion_texture_srv = RAL_NULL_BINDLESS_INDEX;
    return l_material;
}

// Pre-exposure
const static float g_max_half_float = 65504.0f;
const static float g_pre_exposure = 1024.0f;

float3 clamp_to_half_float_range(float3 p_val)
{
    return clamp(p_val, float(0.0f).xxx, g_max_half_float.xxx);
}

// Apply pre-exposure as a mean to prevent compression overflow
float3 pack_lighting(float3 p_color)
{
#if MB_HDR_USE_FLOAT32
    return p_color;
#else
    p_color = p_color / g_pre_exposure;
    return clamp_to_half_float_range(p_color);
#endif
}

float3 unpack_lighting(float3 p_color)
{
#if MB_HDR_USE_FLOAT32
    return p_color;
#else
    return p_color * g_pre_exposure;
#endif
}

// Run-time TBN generation
// Source: http://www.thetenthplanet.de/archives/1180
float3x3 build_tbn(
    float3 p_normal,
    float2 p_uv
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    , float3 p_position_ddx
    , float3 p_position_ddy
    , float2 p_uv_ddx
    , float2 p_uv_ddy
#else
    , float3 p_position
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    )
{
    // Get edge vectors of the pixel triangle
#ifdef MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE
    float3 l_dpdx = p_position_ddx;
    float3 l_dpdy = p_position_ddy;
    float2 l_dudx = p_uv_ddx;
    float2 l_dudy = p_uv_ddy;
#else
    float3 l_dpdx = ddx(p_position);
    float3 l_dpdy = ddy(p_position);
    float2 l_dudx = ddx(p_uv);
    float2 l_dudy = ddy(p_uv);
#endif //MB_COMPUTE_MANUAL_PARTIAL_DERIVATIVE

    // Solve the linear system
    float3 l_dpdy_perp = cross(l_dpdy, p_normal);
    float3 l_dpdx_perp = cross(p_normal, l_dpdx);
    float3 l_t = l_dpdy_perp * l_dudx.x + l_dpdx_perp * l_dudy.x;
    float3 l_b = l_dpdy_perp * l_dudx.y + l_dpdx_perp * l_dudy.y;

    // Construct a scale-invariant frame
    float invmax = rsqrt(max(dot(l_t,l_t), dot(l_b,l_b)));
    return float3x3(l_t * invmax, -l_b * invmax, p_normal);
}

// Multiply float4x3 matrices assuming they are world matricies
float4x3 mul_world_matricies(in float4x3 p_a, in float4x3 p_b)
{
    float4x3 l_result;
    l_result._11  = p_a._11 * p_b._11 + p_a._12 * p_b._21 + p_a._13 * p_b._31;
    l_result._12  = p_a._11 * p_b._12 + p_a._12 * p_b._22 + p_a._13 * p_b._32;
    l_result._13  = p_a._11 * p_b._13 + p_a._12 * p_b._23 + p_a._13 * p_b._33;
    l_result._21  = p_a._21 * p_b._11 + p_a._22 * p_b._21 + p_a._23 * p_b._31;
    l_result._22  = p_a._21 * p_b._12 + p_a._22 * p_b._22 + p_a._23 * p_b._32;
    l_result._23  = p_a._21 * p_b._13 + p_a._22 * p_b._23 + p_a._23 * p_b._33;
    l_result._31  = p_a._31 * p_b._11 + p_a._32 * p_b._21 + p_a._33 * p_b._31;
    l_result._32  = p_a._31 * p_b._12 + p_a._32 * p_b._22 + p_a._33 * p_b._32;
    l_result._33  = p_a._31 * p_b._13 + p_a._32 * p_b._23 + p_a._33 * p_b._33;
    l_result._41  = p_a._41 * p_b._11 + p_a._42 * p_b._21 + p_a._43 * p_b._31 + p_b._41;
    l_result._42  = p_a._41 * p_b._12 + p_a._42 * p_b._22 + p_a._43 * p_b._32 + p_b._42;
    l_result._43  = p_a._41 * p_b._13 + p_a._42 * p_b._23 + p_a._43 * p_b._33 + p_b._43;
    return l_result;
}

// Bounding sphere vs frustum
bool is_sphere_inside_camera_frustum(in cb_camera_t p_camera, in float4 p_bounding_sphere)
{
    for(uint32_t l_plane_index = 0; l_plane_index < p_camera.m_num_frustum_planes; ++l_plane_index)
    {
        if (dot(p_bounding_sphere.xyz, p_camera.m_frustum_planes[l_plane_index].xyz) + p_camera.m_frustum_planes[l_plane_index].w + p_bounding_sphere.w < 0)
        {
            return false;
        }
    }

    return true;
}

// p_proj_pos_curr - Current frame clip-space position (X,Y,W)
// p_proj_pos_prev - Previous frame clip-space position (X,Y,W)
float2 get_motion_vector_without_jitter(float2 p_screen_size, float3 p_proj_pos_curr, float3 p_proj_pos_prev, float2 p_jitter, float2 p_jitter_prev)
{
    float2 l_pos_curr = p_proj_pos_curr.xy / p_proj_pos_curr.z;
    float2 l_pos_prev = p_proj_pos_prev.xy / p_proj_pos_prev.z;

    float2 l_motion_vector = l_pos_prev - l_pos_curr; // FSR expects prev - curr
    l_motion_vector *= float2(0.5f, -0.5f); // NDC space to motion vector space (+Y is top-down)
    l_motion_vector = l_motion_vector * p_screen_size - p_jitter_prev + p_jitter;

    return l_motion_vector;
}

float3 pos_to_uv_depth(float3 p_pos, float4x4 p_mat)
{
    float4 l_pos_projection_space = mul(float4(p_pos, 1.f), p_mat);
    float3 l_pos_ndc = l_pos_projection_space.xyz / l_pos_projection_space.w;
    l_pos_ndc.y = -l_pos_ndc.y;

    return float3(l_pos_ndc.xy * .5f + .5f, l_pos_ndc.z);
}

float3 uv_depth_to_pos(float2 p_uv, float p_depth, float4x4 p_inverse_mat)
{
    float3 l_pos_ndc = float3(p_uv * 2.f - 1.f, p_depth);
    l_pos_ndc.y = -l_pos_ndc.y;

    float4 l_pos_projection_space = float4(l_pos_ndc, 1.0f);
    float4 l_pos_view_space = mul(l_pos_projection_space, p_inverse_mat);
    return l_pos_view_space.xyz / l_pos_view_space.w;
}

float3 get_world_space_local_position(float2 p_uv, float p_depth, float4x4 p_inv_view_proj_local)
{
    float2 l_pos_cs = (p_uv * 2 - 1) ;

    float3 l_pos_ws_local = l_pos_cs.x * p_inv_view_proj_local[0].xyz +
                           -l_pos_cs.y * p_inv_view_proj_local[1].xyz +
                            p_inv_view_proj_local[3].xyz;

    return l_pos_ws_local / (p_depth * p_inv_view_proj_local[2].w + p_inv_view_proj_local[3].w);
}

// For regular z: Zview = near * far / ((near - far) * Zndc + far)
// For reversed z: Zview = near * far / ((far - near) * Zndc + near)
// We are using reversed z.
float get_view_depth_from_depth(float p_depth, float p_near, float p_far)
{
    float l_view_depth = p_near * p_far / ((p_far - p_near) * p_depth + p_near);
    return l_view_depth;
}

float2 get_tangent_half_fov_from_projection_matrix(float4x4 p_projection_matrix)
{
    return float2(1.0 / abs(p_projection_matrix[0][0]), 1.0 / abs(p_projection_matrix[1][1]));
}

// Get view position from view depth
float3 get_view_position(float2 p_uv, float p_view_depth, float2 p_tan_half_fov_xy)
{
    float3 l_position_vs;
    l_position_vs.z = p_view_depth;
    l_position_vs.x = (p_uv.x * 2.0f - 1.0f) * p_view_depth * p_tan_half_fov_xy.x;
    l_position_vs.y = -(p_uv.y * 2.0f - 1.0f) * p_view_depth * p_tan_half_fov_xy.y;
    return l_position_vs;
}

// Wave Intrinsics
#define MB_SCALARIZE_START(expr)                                                            \
    {                                                                                       \
        uint4 l_lane_ballot = current_lane_ballot();                                        \
        uint4 l_active_ballot = WaveActiveBallot(true);                                     \
        [loop]                                                                              \
        while(any((l_lane_ballot & l_active_ballot) != 0))                                  \
        {                                                                                   \
            uint l_wave_first_expr = WaveReadLaneFirst(expr);                               \
            uint4 l_active_thread_mask = WaveActiveBallot(expr == l_wave_first_expr);       \
            l_active_ballot &= ~l_active_thread_mask;                                       \
            [branch]                                                                        \
            if(expr == l_wave_first_expr)                                                   \
            {

#define MB_SCALARIZE_END        \
            }                   \
        }                       \
    }

uint4 current_lane_ballot()
{
    uint4 l_lane_ballot = (uint4)0;
    uint l_thread_idx = WaveGetLaneIndex();
    uint l_idx = l_thread_idx >> 5;         // division by 32
    uint l_subidx = l_thread_idx & 31;      // module by 32

    l_lane_ballot[l_idx] = 1u << l_subidx;

    return l_lane_ballot;
}

// Source: https://github.com/GPUOpen-Effects/FidelityFX-Denoiser/blob/master/ffx-shadows-dnsr/ffx_denoiser_shadows_util.h
//  LANE TO 8x8 MAPPING
//  ===================
//  00 01 08 09 10 11 18 19
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f
//  20 21 28 29 30 31 38 39
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f

uint bitfield_extract(uint src, uint off, uint bits) { uint mask = (1u << bits) - 1; return (src >> off) & mask; } // ABfe
uint bitfield_insert(uint src, uint ins, uint bits) { uint mask = (1u << bits) - 1; return (ins & mask) | (src & (~mask)); } // ABfiM

uint2 remap_lane_8x8(uint lane)
{
    return uint2(
        bitfield_insert(bitfield_extract(lane, 2u, 3u), lane                            , 1u),
        bitfield_insert(bitfield_extract(lane, 3u, 3u), bitfield_extract(lane, 1u, 2u)  , 2u));
}

struct mesh_vertex_t
{
    float3 m_position;
    float3 m_normal;
    float2 m_uv0;
    float4 m_tangent;
};

uint get_vertex_mesh_index(uint p_vertex_index, sb_render_item_t p_render_item)
{
    ByteAddressBuffer l_index_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_index_buffer_srv)];
    uint l_index = 0;
    uint l_byte_offset = p_render_item.m_index_buffer_offset + p_render_item.m_index_buffer_stride * p_vertex_index;
    if (p_render_item.m_index_buffer_stride == sizeof(uint))
    {
        l_index = l_index_mesh.Load<uint>(l_byte_offset);
    }
    else if (p_render_item.m_index_buffer_stride == sizeof(min16uint))
    {
        // Cannot fetch 16bit types from buffer
        // Apply manual unpacking using 32bit types
        l_index = l_index_mesh.Load<uint>(4 * (l_byte_offset / 4));
        l_index = ((l_byte_offset / 2) & 1) == 0 ? l_index & 0xFFFF : l_index >> 16;
    }
    else if (p_render_item.m_index_buffer_stride == 1) // byte
    {
        // Cannot fetch 8bit types from buffer
        // Apply manual unpacking using 32bit types
        l_index = l_index_mesh.Load<uint>(4 * (l_byte_offset / 4));
        //l_index = l_byte_offset % 4 == 0 ? l_index & 0xFFFF : l_index >> 16;
        switch (l_byte_offset & 3)
        {
            case 0: l_index = (l_index & 0x000000FF); break;
            case 1: l_index = (l_index & 0x0000FF00) >> 8; break;
            case 2: l_index = (l_index & 0x00FF0000) >> 16; break;
            case 3: l_index = (l_index & 0xFF000000) >> 24; break;
        }
    }

    return l_index;
}

void get_vertex_mesh_position(uint p_index, sb_render_item_t p_render_item, out mesh_vertex_t p_result)
{
    // Position
    ByteAddressBuffer l_position_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_position_buffer_srv)];
    if (p_render_item.m_position_stride == sizeof(float3))
    {
        p_result.m_position = l_position_mesh.Load<float3>(p_render_item.m_position_offset + p_index * p_render_item.m_position_stride);
    }
    else if (p_render_item.m_position_stride == sizeof(float2))
    {
        p_result.m_position.xy = l_position_mesh.Load<float2>(p_render_item.m_position_offset + p_index * p_render_item.m_position_stride);
        p_result.m_position.z = 0;
    }
}

void get_vertex_mesh_other(uint p_index, sb_render_item_t p_render_item, out mesh_vertex_t p_result)
{
    // Normal
    ByteAddressBuffer l_normal_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_normal_buffer_srv)];
    if (p_render_item.m_normal_stride == sizeof(float3))
    {
        p_result.m_normal = l_normal_mesh.Load<float3>(p_render_item.m_normal_offset + p_index * p_render_item.m_normal_stride);
    }

    // UV
    ByteAddressBuffer l_uv0_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_uv0_buffer_srv)];
    if (p_render_item.m_uv0_stride == sizeof(float2))
    {
        p_result.m_uv0 = l_uv0_mesh.Load<float2>(p_render_item.m_uv0_offset + p_index * p_render_item.m_uv0_stride);
    }

    // Tangent
    ByteAddressBuffer l_tangent_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_tangent_buffer_srv)];
    if (p_render_item.m_tangent_stride == sizeof(float3))
    {
        p_result.m_tangent = float4(l_tangent_mesh.Load<float3>(p_render_item.m_tangent_offset + p_index * p_render_item.m_tangent_stride), 1.0f);
    }
    else if (p_render_item.m_tangent_stride == sizeof(float4))
    {
        p_result.m_tangent = l_tangent_mesh.Load<float4>(p_render_item.m_tangent_offset + p_index * p_render_item.m_tangent_stride);
    }
}

//! \param p_vertex_mesh_index Data index in the position buffer.
//! \param p_extra_offset Extra offset in case we have multiple instances in the buffer.
void get_vertex_mesh_with_index(uint p_vertex_mesh_index, sb_render_item_t p_render_item, uint p_extra_offset, out mesh_vertex_t p_result)
{
    // Position
    ByteAddressBuffer l_position_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_position_buffer_srv)];
    if (p_render_item.m_position_stride == sizeof(float3))
    {
        p_result.m_position = l_position_mesh.Load<float3>(p_extra_offset * p_render_item.m_position_stride + p_render_item.m_position_offset + p_vertex_mesh_index * p_render_item.m_position_stride);
    }
    else if (p_render_item.m_position_stride == sizeof(float2))
    {
        p_result.m_position.xy = l_position_mesh.Load<float2>(p_extra_offset * p_render_item.m_position_stride + p_render_item.m_position_offset + p_vertex_mesh_index * p_render_item.m_position_stride);
        p_result.m_position.z = 0;
    }

    // Normal
    ByteAddressBuffer l_normal_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_normal_buffer_srv)];
    if (p_render_item.m_normal_stride == sizeof(float3))
    {
        p_result.m_normal = l_normal_mesh.Load<float3>(p_extra_offset * p_render_item.m_normal_stride + p_render_item.m_normal_offset + p_vertex_mesh_index * p_render_item.m_normal_stride);
    }

    // UV
    ByteAddressBuffer l_uv0_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_uv0_buffer_srv)];
    if (p_render_item.m_uv0_stride == sizeof(float2))
    {
        p_result.m_uv0 = l_uv0_mesh.Load<float2>(p_extra_offset * p_render_item.m_uv0_stride + p_render_item.m_uv0_offset + p_vertex_mesh_index * p_render_item.m_uv0_stride);
    }

    // Tangent
    ByteAddressBuffer l_tangent_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_tangent_buffer_srv)];
    if (p_render_item.m_tangent_stride == sizeof(float3))
    {
        p_result.m_tangent = float4(l_tangent_mesh.Load<float3>(p_extra_offset * p_render_item.m_tangent_stride + p_render_item.m_tangent_offset + p_vertex_mesh_index * p_render_item.m_tangent_stride), 1.0f);
    }
    else if (p_render_item.m_tangent_stride == sizeof(float4))
    {
        p_result.m_tangent = l_tangent_mesh.Load<float4>(p_extra_offset * p_render_item.m_tangent_stride + p_render_item.m_tangent_offset + p_vertex_mesh_index * p_render_item.m_tangent_stride);
    }
}

//! \param p_vertex_index Data index in the index buffer.
//! \param p_extra_offset Extra offset in case we have multiple instances in the buffer.
void get_vertex_mesh(uint p_vertex_index, sb_render_item_t p_render_item, uint p_extra_offset, out mesh_vertex_t p_result)
{
    // Index buffer
    uint l_index = get_vertex_mesh_index(p_vertex_index, p_render_item);

    get_vertex_mesh_with_index(l_index, p_render_item, p_extra_offset, p_result);
}

//-----------------------------------------------------------------------------
bool screen_coverage_test(float4x4 p_proj, float p_fov_vertical, float4 p_bounding_sphere, float p_distance_z, float2 p_threshold_range)
{
    // Different logic depending on orthographic or perspective camera
    bool l_is_orthographic = p_proj._44 == 1.0f;
    float l_screen_coverage = l_is_orthographic ? p_bounding_sphere.w * max(p_proj._11, p_proj._22) : p_bounding_sphere.w / (p_distance_z * p_fov_vertical);
    return p_threshold_range.x < l_screen_coverage && l_screen_coverage <= p_threshold_range.y;
}

//-----------------------------------------------------------------------------
float4 get_bounding_sphere_ws(float4 p_bounding_sphere_model_space, float4x3 p_transform, float p_culling_scale)
{
    float3 l_bounding_sphere_center = mul(p_bounding_sphere_model_space.xyz, (float3x3)p_transform) + p_transform._41_42_43;
    float3 l_transform_scale = float3(length(p_transform._11_12_13), length(p_transform._21_22_23), length(p_transform._31_32_33));
    float l_transform_max_scale = max(l_transform_scale.x, max(l_transform_scale.y, l_transform_scale.z));
    l_transform_max_scale = max(l_transform_max_scale, p_culling_scale);
    float4 l_bounding_sphere = float4(l_bounding_sphere_center, l_transform_max_scale * p_bounding_sphere_model_space.w);
    return l_bounding_sphere;
}

float4 get_bounding_sphere_ms(float4 p_bounding_sphere_model_space, float p_culling_scale)
{
    return float4(p_bounding_sphere_model_space.xyz, p_bounding_sphere_model_space.w * p_culling_scale);
}

#endif // MB_SHADER_COMMON_HLSL
