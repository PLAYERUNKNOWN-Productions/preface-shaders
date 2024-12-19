// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

// PBR
#include "mb_lighting_common.hlsl"

// Quadtree
#include "../helper_shaders/mb_quadtree_common.hlsl"

// Atmospheric scattering
#include "mb_atmospheric_scattering_utils.hlsl"

// Path tracing defines
#define MIP_LEVEL 0
#define RUSSIAN_ROULETTE_MIN_BOUNCE 3
#define NO_VT_FEEDBACK 100000
//#define USE_HAMMERSLEY 1
//#define USE_UNIFORM_SAMPLING 1

//#define DEBUG_NO_MATERIALS 1

#if defined(MB_RAYTRACING_DIFFUSE_GI)
    #define MAX_STANDARD_RAY_DISTANCE 1000
    #define MAX_SHADOW_RAY_DISTANCE 1000
    #define ATMOSPHERIC_SCATTERING_ON_MISS
#else
    #define MAX_STANDARD_RAY_DISTANCE 2000000
    #define MAX_SHADOW_RAY_DISTANCE 10000
    #define ATMOSPHERIC_SCATTERING_ON_HIT
    #define ATMOSPHERIC_SCATTERING_ON_MISS
#endif
// Ray indices: used in TraceRay(...)
#define STANDARD_RAY_INDEX  0
#define SHADOW_RAY_INDEX    1

// Atmospheric scattering
#define SKY_DISK_ENABLED 1.0f
#define SKY_DISK_DISABLED 0

// Super sampling
#define CAMERA_SUPERSAMPLING

// Sun as a disk
#define SUN_ANGLE_RADIUS (0.5f * 0.53f * 3.14f / 180.0f)

// Diffuse GI: num jittered positions
#define MB_DIFFUSE_GI_NUM_JITTERED_POSITIONS 64

// Temporary define to match with reference  path tracing
//#define DEBUG_PROCESSING 1

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_raytracing_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct raytracing_mesh_vertex_t
{
    float3 m_position;
    float3 m_normal;
    float3 m_tangent;
    float3 m_binormal;
    float2 m_uv0;
};

//-----------------------------------------------------------------------------
struct pbr_material_t
{
    float3  m_base_color;
    float3  m_diffuse_reflectance;
    float3  m_specular_f0;

    float3  m_emissive;
    float   m_roughness;
    float   m_metallic;
    float   m_ior;
    float3  m_normal_ts;

    float   m_opacity;
    float   m_alpha_cutoff;
};

//-----------------------------------------------------------------------------
struct downsampled_data_t
{
    // Downsampled depth
    float m_depth;

    // Downsampled velocity
    float2 m_velocity;

    // Offsets in pixels from low-res sample corner to the sample that was selected
    uint2 m_offset_in_pixels;
};

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
rand_type_t hash_wang(inout rand_type_t p_seed)
{
    p_seed = uint(p_seed ^ uint(61)) ^ uint(p_seed >> uint(16));
    p_seed *= uint(9);
    p_seed = p_seed ^ (p_seed >> 4);
    p_seed *= uint(0x27d4eb2d);
    p_seed = p_seed ^ (p_seed >> 15);
    return p_seed;
}

//-----------------------------------------------------------------------------
// 32-bit Xorshift random number generator
uint xorshift(inout uint p_rand_state)
{
    p_rand_state ^= p_rand_state << 13;
    p_rand_state ^= p_rand_state >> 17;
    p_rand_state ^= p_rand_state << 5;
    return p_rand_state;
}

//-----------------------------------------------------------------------------
// Jenkins's "one at a time" hash function
uint jenkins_hash(uint p_val)
{
    p_val += p_val << 10;
    p_val ^= p_val >> 6;
    p_val += p_val << 3;
    p_val ^= p_val >> 11;
    p_val += p_val << 15;
    return p_val;
}

//-----------------------------------------------------------------------------
// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uint_to_float(uint x)
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

#if defined(USE_RAND_XORSHIFT)
//-----------------------------------------------------------------------------
rand_type_t init_rand(uint2 p_pixel_coords, uint2 p_resolution, uint p_frame_number)
{
    rand_type_t l_seed = dot(p_pixel_coords, uint2(1, p_resolution.x)) ^ jenkins_hash(p_frame_number);
    return jenkins_hash(l_seed);
}

float random_float01(inout rand_type_t p_state)
{
    return uint_to_float(xorshift(p_state));
}
#else
//-----------------------------------------------------------------------------
rand_type_t init_rand(uint2 p_pixel_coords, uint2 p_resolution, uint p_frame_number)
{
    rand_type_t l_seed = uint(uint(p_pixel_coords.x) * uint(1973) + uint(p_pixel_coords.y) * uint(9277) + uint(p_frame_number) * uint(26699)) | uint(1);
    return l_seed;
}

float random_float01(inout rand_type_t p_state)
{
    return float(hash_wang(p_state)) / 4294967296.0;
}
#endif

//-----------------------------------------------------------------------------
float2 sample_concentric_disk(float2 p_u)
{
    float2 l_u_offfset = 2.0f * p_u - float2(1.0f, 1.0f);

    // Handle degeneracy at the origin
    if (l_u_offfset.x == 0 && l_u_offfset.y == 0)
        return float2(0, 0);

    // Apply concentric mapping to point
    float l_theta;
    float l_r;
    if (abs(l_u_offfset.x) > abs(l_u_offfset.y))
    {
        l_r = l_u_offfset.x;
        l_theta = (M_PI / 4.0f) * (l_u_offfset.y / l_u_offfset.x);
    } else {
        l_r = l_u_offfset.y;
        l_theta = (M_PI / 2.0f) - (M_PI / 4.0f) * (l_u_offfset.x / l_u_offfset.y);
    }
    return l_r * float2(cos(l_theta), sin(l_theta));
}

//-----------------------------------------------------------------------------
float3 sample_hemisphere_uniform(float p_u, float p_v)
{
    float l_z = p_u;
    float l_r = sqrt(1.0f - l_z * l_z);
    float l_phi = 2.0f * M_PI * p_v;
    return float3(l_r * cos(l_phi), l_r * sin(l_phi), l_z);
}

//-----------------------------------------------------------------------------
float3 sample_hemisphere_cosine(float p_u, float p_v)
{
    float2 l_d = sample_concentric_disk(float2(p_u, p_v));
    float l_z = sqrt(max(0, 1.0f - l_d.x * l_d.x - l_d.y * l_d.y));
    return float3(l_d.x, l_d.y, l_z);
}

//-----------------------------------------------------------------------------
// Samples a microfacet normal for the GGX distribution using VNDF method.
// Source: "Sampling the GGX Distribution of Visible Normals" by Heitz
// See also https://hal.inria.fr/hal-00996995v1/document and http://jcgt.org/published/0007/04/01/
// Random variables 'u' must be in <0;1) interval
// PDF is 'G1(NdotV) * D'
float3 sample_ggx_visible_normal(float3 p_view, float p_alpha_x, float p_alpha_y, float p_rand0, float p_rand1)
{
    // Stretch the view vector so we are sampling as though
    // roughness==1
    float3 l_view = normalize(float3(p_view.x * p_alpha_x, p_view.y * p_alpha_y, p_view.z));

    // Build an orthonormal basis with l_view, t1, and t2
    float3 l_t1 = (l_view.z < 0.999f) ? normalize(cross(l_view, float3(0, 0, 1))) : float3(1, 0, 0);
    float3 l_t2 = cross(l_t1, l_view);

    // Choose a point on a disk with each half of the disk weighted
    // proportionally to its projection onto direction l_view
    float l_a = 1.0f / (1.0f + l_view.z);
    float l_r = sqrt(p_rand0);
    float l_phi = (p_rand1 < l_a) ? (p_rand1 / l_a) * M_PI : M_PI + (p_rand1 - l_a) / (1.0f - l_a) * M_PI;
    float l_p1 = l_r * cos(l_phi);
    float l_p2 = l_r * sin(l_phi) * ((p_rand1 < l_a) ? 1.0f : l_view.z);

    // Calculate the normal in this stretched tangent space
    float3 l_normal = l_p1 * l_t1 + l_p2 * l_t2 + sqrt(max(0.0f, 1.0f - l_p1 * l_p1 - l_p2 * l_p2)) * l_view;

    // Unstretch and normalize the normal
    return normalize(float3(p_alpha_x * l_normal.x, p_alpha_y * l_normal.y, max(0.0f, l_normal.z)));
}

//-----------------------------------------------------------------------------
// Offsets the ray origin from current position p_position, along normal p_geo_normal (which must be geometric normal) so that no self-intersection can occur.
float3 offset_ray(const float3 p_position, const float3 p_geo_normal)
{
    static const float l_origin = 1.0f / 32.0f;
    static const float l_float_scale = 1.0f / 65536.0f;
    static const float l_int_scale = 256.0f;

    int3 of_i = int3(l_int_scale * p_geo_normal.x, l_int_scale * p_geo_normal.y, l_int_scale * p_geo_normal.z);

    float3 p_i = float3(asfloat(asint(p_position.x) + ((p_position.x < 0) ? -of_i.x : of_i.x)),
                        asfloat(asint(p_position.y) + ((p_position.y < 0) ? -of_i.y : of_i.y)),
                        asfloat(asint(p_position.z) + ((p_position.z < 0) ? -of_i.z : of_i.z)));

    return float3(  abs(p_position.x) < l_origin ? p_position.x + l_float_scale * p_geo_normal.x : p_i.x,
                    abs(p_position.y) < l_origin ? p_position.y + l_float_scale * p_geo_normal.y : p_i.y,
                    abs(p_position.z) < l_origin ? p_position.z + l_float_scale * p_geo_normal.z : p_i.z);
}

//-----------------------------------------------------------------------------
raytracing_mesh_vertex_t get_vertex_mesh(sb_raytracing_hierarchy_t p_hierarchy,
                                         uint p_geo_index,
                                         uint p_triangle_index,
                                         uint p_vertex_index)
{
    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[p_hierarchy.m_render_item_id];

    // Init output
    raytracing_mesh_vertex_t l_mesh_vertex = (raytracing_mesh_vertex_t)0;

    // Index buffer
    ByteAddressBuffer l_index_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_index_buffer_srv)];
    uint l_index = 0;
    uint l_byte_offset = l_render_item.m_index_buffer_offset + l_render_item.m_index_buffer_stride * (p_triangle_index * 3 + p_vertex_index);
    if (l_render_item.m_index_buffer_stride == sizeof(uint))
    {
        l_index = l_index_mesh.Load<uint>(l_byte_offset);
    }
    else if (l_render_item.m_index_buffer_stride == sizeof(min16uint))
    {
        // Cannot fetch 16bit types from buffer
        // Apply manual unpacking using 32bit types
        l_index = l_index_mesh.Load<uint>(4 * (l_byte_offset / 4));
        l_index = (l_byte_offset / 2) % 2 == 0 ? l_index & 0xFFFF : l_index >> 16;
    }
    else
    {
        return l_mesh_vertex;
    }

    // Position
    ByteAddressBuffer l_position_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_position_buffer_srv)];
    if (l_render_item.m_position_stride == sizeof(float3))
    {
        l_mesh_vertex.m_position = l_position_mesh.Load<float3>(l_render_item.m_position_offset + l_index * l_render_item.m_position_stride);
    }

    // Normal
    ByteAddressBuffer l_normal_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_normal_buffer_srv)];
    if (l_render_item.m_normal_stride == sizeof(float3))
    {
        l_mesh_vertex.m_normal = l_normal_mesh.Load<float3>(l_render_item.m_normal_offset + l_index * l_render_item.m_normal_stride);
    }

    // Tangent
    float4 l_tangent = 0;
    ByteAddressBuffer l_tangent_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_tangent_buffer_srv)];
    if (l_render_item.m_tangent_stride == sizeof(float3))
    {
        l_tangent = float4(l_tangent_mesh.Load<float3>(l_render_item.m_tangent_offset + l_index * l_render_item.m_tangent_stride), 1.0f);
    }
    else if (l_render_item.m_tangent_stride == sizeof(float4))
    {
        l_tangent = l_tangent_mesh.Load<float4>(l_render_item.m_tangent_offset + l_index * l_render_item.m_tangent_stride);
    }
    l_mesh_vertex.m_tangent = l_tangent.xyz;

    // Binormal
    l_mesh_vertex.m_binormal = cross(l_mesh_vertex.m_tangent, l_mesh_vertex.m_normal) * l_tangent.w;

    // UV
    ByteAddressBuffer l_uv0_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_uv0_buffer_srv)];
    if (l_render_item.m_uv0_stride == sizeof(float2))
    {
        l_mesh_vertex.m_uv0 = l_uv0_mesh.Load<float2>(l_render_item.m_uv0_offset + l_index * l_render_item.m_uv0_stride);
    }

    return l_mesh_vertex;
}

//-----------------------------------------------------------------------------
raytracing_mesh_vertex_t get_vertex_mesh_tile(sb_render_item_t p_render_item,
                                              uint p_triangle_index,
                                              sb_quadtree_material_t p_quadtree_material,
                                              uint p_vertex_index)
{
    // Init output
    raytracing_mesh_vertex_t l_mesh_vertex = (raytracing_mesh_vertex_t)0;

    // Index buffer
    ByteAddressBuffer l_index_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_index_buffer_srv)];
    uint l_index = 0;
    uint l_byte_offset = p_render_item.m_index_buffer_offset + p_render_item.m_index_buffer_stride * (p_triangle_index * 3 + p_vertex_index);
    if (p_render_item.m_index_buffer_stride == sizeof(uint))
    {
        l_index = l_index_mesh.Load<uint>(l_byte_offset);
    }
    else if (p_render_item.m_index_buffer_stride == sizeof(min16uint))
    {
        // Cannot fetch 16bit types from buffer
        // Apply manual unpacking using 32bit types
        l_index = l_index_mesh.Load<uint>(4 * (l_byte_offset / 4));
        l_index = (l_byte_offset / 2) % 2 == 0 ? l_index & 0xFFFF : l_index >> 16;
    }
    else
    {
        return l_mesh_vertex;
    }

    // Position
    ByteAddressBuffer l_position_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_position_buffer_srv)];
    if (p_render_item.m_position_stride == sizeof(float3))
    {
        l_mesh_vertex.m_position = l_position_mesh.Load<float3>(p_render_item.m_position_offset + l_index * p_render_item.m_position_stride);
    }

    // Normal
    ByteAddressBuffer l_normal_mesh = ResourceDescriptorHeap[NonUniformResourceIndex(p_render_item.m_normal_buffer_srv)];
    if (p_render_item.m_normal_stride == sizeof(float3))
    {
        l_mesh_vertex.m_normal = l_normal_mesh.Load<float3>(p_render_item.m_normal_offset + l_index * p_render_item.m_normal_stride);
    }

    // For tile meshes this will be computed per triangle during interpolation stage
    l_mesh_vertex.m_tangent = 0;
    l_mesh_vertex.m_binormal = 0;

    // Tile UV is calculated from vertex index
    uint l_vertex_resolution = p_quadtree_material.m_tile_size_in_vertices;
    uint l_tile_position_x = l_index % l_vertex_resolution;
    uint l_tile_position_y = (l_index - l_tile_position_x) / l_vertex_resolution;
    l_mesh_vertex.m_uv0 = float2(l_tile_position_x / (float) (l_vertex_resolution - 1), l_tile_position_y / (float) (l_vertex_resolution - 1));

    return l_mesh_vertex;
}

//-----------------------------------------------------------------------------
// TODO: copy-pasted
float3 ggx_specular_lighting(float3 p_normal, float3 p_view_dir, float3 p_light_dir, float3 p_specular_f0, float p_roughness)
{
    // Prepare vectors
    float3 l_half = normalize(p_light_dir + p_view_dir);
    float l_ndl = max(0.001f, min(1.0f, dot(p_normal, p_light_dir)));
    float l_ndv = max(0.001f, min(1.0f, (dot(p_normal, p_view_dir))));
    float l_ndh = max(0.001f, min(1.0f, dot(p_normal, l_half)));
    float l_vdh = max(0.001f, min(1.0f, dot(p_view_dir, l_half)));

    // Clamp and remap roughness
    float l_alpha = clamp_and_remap_roughness(p_roughness);

    // Normal distribution function
    float l_d = distribution_ggx(l_ndh, l_alpha);

    // Visibility function
    float l_g = vis_smith_joint(l_ndv, l_ndl, l_alpha);

    // Fresnel equation
    float3 l_f = fresnel_reflectance_schlick(p_specular_f0, l_vdh);

    // Cook-Torrance BRDF
    float3 l_specular_lighting = l_d * l_g * l_f;

    return l_specular_lighting * l_ndl;
}

//-----------------------------------------------------------------------------
float smith_g1(float3 p_normal, float3 p_light, float3 p_view, float p_alpha2)
{
    float l_ndl = max(0.001f, min(1.0f, dot(p_normal, p_light)));
    float l_ndv = max(0.001f, min(1.0f, dot(p_normal, p_view)));
    float l_denom_c = sqrt(p_alpha2 + (1.0f - p_alpha2) * l_ndv * l_ndv) + l_ndv;

    return 2.0f * l_ndv / l_denom_c;
}

//-----------------------------------------------------------------------------
float smith_g2(float3 p_normal, float3 p_light, float3 p_view, float p_alpha2)
{
    float l_ndl = max(0.001f, min(1.0f, dot(p_normal, p_light)));
    float l_ndv = max(0.001f, min(1.0f, dot(p_normal, p_view)));

    float l_denom_a = l_ndv * sqrt(p_alpha2 + (1.0f - p_alpha2) * l_ndl * l_ndl);
    float l_denom_b = l_ndl * sqrt(p_alpha2 + (1.0f - p_alpha2) * l_ndv * l_ndv);

    return 2.0f * l_ndl * l_ndv / (l_denom_a + l_denom_b);
}

//-----------------------------------------------------------------------------
float get_specular_brdf_probability(float3 p_diffuse_reflectance,
                                    float3 p_specular_f0,
                                    float3 p_view_dir,
                                    float3 p_normal)
{
    float l_vdn = max(0.001f, min(1.0f, dot(p_view_dir, p_normal)));

    // Fresnel equation
    // Note: we use the shading normal instead of the microfacet normal (half-vector) for Fresnel term here
    // Half-vector is yet unknown at this point
    float3 l_fresnel = fresnel_reflectance_schlick(p_specular_f0, l_vdn);
    float l_fresnel_luminance = get_luminance(l_fresnel_luminance);

    // Approximate relative specular and diffuse contribution
    float l_spec_contribution = l_fresnel_luminance;
    float l_diff_contribution = get_luminance(p_diffuse_reflectance) * (1.0f - l_fresnel_luminance);

    // Return the probability of selecting specular BRDF over diffuse BRDF
    float l_probability = l_spec_contribution / max(0.0001f, l_spec_contribution + l_diff_contribution);

    // Clamp probability to avoid undersampling
    return clamp(l_probability, 0.1f, 0.9f);
}

//-----------------------------------------------------------------------------
raytracing_mesh_vertex_t get_vertex_mesh(sb_raytracing_hierarchy_t p_hierarchy,
                                         uint p_geo_index,
                                         uint p_triangle_index,
                                         float3 p_barycentrics)
{
    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[p_hierarchy.m_render_item_id];

    // Get object to world transform
    float4x3 l_object_to_world_4x3 = ObjectToWorld4x3();

    // Get mesh vertex
    raytracing_mesh_vertex_t l_mesh_vertex_0 = get_vertex_mesh(p_hierarchy, p_geo_index, p_triangle_index, 0);
    raytracing_mesh_vertex_t l_mesh_vertex_1 = get_vertex_mesh(p_hierarchy, p_geo_index, p_triangle_index, 1);
    raytracing_mesh_vertex_t l_mesh_vertex_2 = get_vertex_mesh(p_hierarchy, p_geo_index, p_triangle_index, 2);

    //
    raytracing_mesh_vertex_t l_mesh_vertex;
    l_mesh_vertex.m_position =  l_mesh_vertex_0.m_position * p_barycentrics.x +
                                l_mesh_vertex_1.m_position * p_barycentrics.y +
                                l_mesh_vertex_2.m_position * p_barycentrics.z;

    l_mesh_vertex.m_normal =    l_mesh_vertex_0.m_normal * p_barycentrics.x +
                                l_mesh_vertex_1.m_normal * p_barycentrics.y +
                                l_mesh_vertex_2.m_normal * p_barycentrics.z;

    l_mesh_vertex.m_tangent =   l_mesh_vertex_0.m_tangent * p_barycentrics.x +
                                l_mesh_vertex_1.m_tangent * p_barycentrics.y +
                                l_mesh_vertex_2.m_tangent * p_barycentrics.z;

    l_mesh_vertex.m_binormal =  l_mesh_vertex_0.m_binormal * p_barycentrics.x +
                                l_mesh_vertex_1.m_binormal * p_barycentrics.y +
                                l_mesh_vertex_2.m_binormal * p_barycentrics.z;

    l_mesh_vertex.m_uv0 =       l_mesh_vertex_0.m_uv0 * p_barycentrics.x +
                                l_mesh_vertex_1.m_uv0 * p_barycentrics.y +
                                l_mesh_vertex_2.m_uv0 * p_barycentrics.z;

    // NORMAL: Transorm to world-space
    l_mesh_vertex.m_normal = mul(l_mesh_vertex.m_normal, (float3x3)l_render_item.m_transform);
    l_mesh_vertex.m_normal = mul(l_mesh_vertex.m_normal, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_normal = normalize(l_mesh_vertex.m_normal);

    // TANGENT: Transorm to world-space
    l_mesh_vertex.m_tangent = mul(l_mesh_vertex.m_tangent, (float3x3)l_render_item.m_transform);
    l_mesh_vertex.m_tangent = mul(l_mesh_vertex.m_tangent, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_tangent = normalize(l_mesh_vertex.m_tangent);

    //! \todo Put in material ecaluation code
    // BINORMAL: Transorm to world-space
    l_mesh_vertex.m_binormal = mul(l_mesh_vertex.m_binormal, (float3x3)l_render_item.m_transform);
    l_mesh_vertex.m_binormal = mul(l_mesh_vertex.m_binormal, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_binormal = normalize(l_mesh_vertex.m_binormal);

    // POSITION: Transform to world-space
    float4 l_position        = float4(l_mesh_vertex.m_position.xyz, 1.0f);
    l_position.xyz           = mul(l_position, l_render_item.m_transform);
    l_mesh_vertex.m_position = mul(l_position, l_object_to_world_4x3);

    return l_mesh_vertex;
}

//-----------------------------------------------------------------------------
raytracing_mesh_vertex_t get_vertex_mesh_tile(sb_render_item_t p_render_item,
                                              uint p_triangle_index,
                                              float3 p_barycentrics,
                                              sb_quadtree_material_t p_quadtree_material)
{
    // Get object to world transform
    float4x3 l_object_to_world_4x3 = ObjectToWorld4x3();

    // Get mesh vertex
    raytracing_mesh_vertex_t l_mesh_vertex_0 = get_vertex_mesh_tile(p_render_item, p_triangle_index, p_quadtree_material, 0);
    raytracing_mesh_vertex_t l_mesh_vertex_1 = get_vertex_mesh_tile(p_render_item, p_triangle_index, p_quadtree_material, 1);
    raytracing_mesh_vertex_t l_mesh_vertex_2 = get_vertex_mesh_tile(p_render_item, p_triangle_index, p_quadtree_material, 2);

    // Get positions
    float3 l_position_1 = l_mesh_vertex_0.m_position;
    float3 l_position_2 = l_mesh_vertex_1.m_position;
    float3 l_position_3 = l_mesh_vertex_2.m_position;

    // Get UV
    float2 l_uv_1 = l_mesh_vertex_0.m_uv0;
    float2 l_uv_2 = l_mesh_vertex_1.m_uv0;
    float2 l_uv_3 = l_mesh_vertex_2.m_uv0;

    //-----------------------------------------------------
    // Solve linear system
    // Source: https://www.cs.upc.edu/~virtual/G/1.%20Teoria/06.%20Textures/Tangent%20Space%20Calculation.pdf

    float l_x1 = l_position_2.x - l_position_1.x;
    float l_x2 = l_position_3.x - l_position_1.x;
    float l_y1 = l_position_2.y - l_position_1.y;
    float l_y2 = l_position_3.y - l_position_1.y;
    float l_z1 = l_position_2.z - l_position_1.z;
    float l_z2 = l_position_3.z - l_position_1.z;

    float l_s1 = l_uv_2.x - l_uv_1.x;
    float l_s2 = l_uv_3.x - l_uv_1.x;
    float l_t1 = l_uv_2.y - l_uv_1.y;
    float l_t2 = l_uv_3.y - l_uv_1.y;

    float l_r = 1.0f / (l_s1 * l_t2 - l_s2 * l_t1);
    float3 l_tangent_tmp = float3((l_t2 * l_x1 - l_t1 * l_x2) * l_r,
                                  (l_t2 * l_y1 - l_t1 * l_y2) * l_r,
                                  (l_t2 * l_z1 - l_t1 * l_z2) * l_r);
    float3 l_bitangent_tmp = float3((l_s1 * l_x2 - l_s2 * l_x1) * l_r,
                                    (l_s1 * l_y2 - l_s2 * l_y1) * l_r,
                                    (l_s1 * l_z2 - l_s2 * l_z1) * l_r);

    // Normals
    float3 l_normal_0 = normalize(cross(l_position_2 - l_position_1, l_position_3 - l_position_1));
    float3 l_normal_1 = normalize(cross(l_position_3 - l_position_2, l_position_1 - l_position_2));
    float3 l_normal_2 = normalize(cross(l_position_1 - l_position_3, l_position_2 - l_position_3));

    float4 l_tangent_0 = float4(normalize(l_tangent_tmp - l_normal_0 * dot(l_normal_0, l_tangent_tmp)),         // Gram-Schmidt orthogonalize
                               (dot(cross(l_normal_0, l_tangent_tmp), l_bitangent_tmp) < 0.0f) ? -1.0f : 1.0f); // Calculate handedness
    float4 l_tangent_1 = float4(normalize(l_tangent_tmp - l_normal_1 * dot(l_normal_1, l_tangent_tmp)),         // Gram-Schmidt orthogonalize
                               (dot(cross(l_normal_1, l_tangent_tmp), l_bitangent_tmp) < 0.0f) ? -1.0f : 1.0f); // Calculate handedness
    float4 l_tangent_2 = float4(normalize(l_tangent_tmp - l_normal_2 * dot(l_normal_2, l_tangent_tmp)),         // Gram-Schmidt orthogonalize
                               (dot(cross(l_normal_2, l_tangent_tmp), l_bitangent_tmp) < 0.0f) ? -1.0f : 1.0f); // Calculate handedness

    float3 l_binormal_0 = cross(l_tangent_0.xyz, l_mesh_vertex_0.m_normal) * l_tangent_0.w;
    float3 l_binormal_1 = cross(l_tangent_1.xyz, l_mesh_vertex_1.m_normal) * l_tangent_1.w;
    float3 l_binormal_2 = cross(l_tangent_2.xyz, l_mesh_vertex_2.m_normal) * l_tangent_2.w;

    // Interpolate attributes
    raytracing_mesh_vertex_t l_mesh_vertex;
    l_mesh_vertex.m_position =  l_mesh_vertex_0.m_position * p_barycentrics.x +
                                l_mesh_vertex_1.m_position * p_barycentrics.y +
                                l_mesh_vertex_2.m_position * p_barycentrics.z;

    l_mesh_vertex.m_normal =    l_mesh_vertex_0.m_normal * p_barycentrics.x +
                                l_mesh_vertex_1.m_normal * p_barycentrics.y +
                                l_mesh_vertex_2.m_normal * p_barycentrics.z;

    l_mesh_vertex.m_tangent =   l_tangent_0.xyz * p_barycentrics.x +
                                l_tangent_1.xyz * p_barycentrics.y +
                                l_tangent_2.xyz * p_barycentrics.z;

    l_mesh_vertex.m_binormal =  l_binormal_0 * p_barycentrics.x +
                                l_binormal_1 * p_barycentrics.y +
                                l_binormal_2 * p_barycentrics.z;

    l_mesh_vertex.m_uv0 =       l_mesh_vertex_0.m_uv0 * p_barycentrics.x +
                                l_mesh_vertex_1.m_uv0 * p_barycentrics.y +
                                l_mesh_vertex_2.m_uv0 * p_barycentrics.z;

    // NORMAL: Transorm to world-space
    l_mesh_vertex.m_normal = mul(l_mesh_vertex.m_normal, (float3x3)p_render_item.m_transform);
    l_mesh_vertex.m_normal = mul(l_mesh_vertex.m_normal, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_normal = normalize(l_mesh_vertex.m_normal);

    // TANGENT: Transorm to world-space
    l_mesh_vertex.m_tangent = mul(l_mesh_vertex.m_tangent, (float3x3)p_render_item.m_transform);
    l_mesh_vertex.m_tangent = mul(l_mesh_vertex.m_tangent, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_tangent = normalize(l_mesh_vertex.m_tangent);

    //! \todo Put in material ecaluation code
    // BINORMAL: Transorm to world-space
    l_mesh_vertex.m_binormal = mul(l_mesh_vertex.m_binormal, (float3x3)p_render_item.m_transform);
    l_mesh_vertex.m_binormal = mul(l_mesh_vertex.m_binormal, (float3x3)l_object_to_world_4x3);
    l_mesh_vertex.m_binormal = normalize(l_mesh_vertex.m_binormal);

    // POSITION: Transform to world-space
    float4 l_position        = float4(l_mesh_vertex.m_position.xyz, 1.0f);
    l_position.xyz           = mul(l_position, p_render_item.m_transform);
    l_mesh_vertex.m_position = mul(l_position, l_object_to_world_4x3);

    return l_mesh_vertex;
}

//-----------------------------------------------------------------------------
uint is_material_vt(sb_geometry_pbr_material_t p_material)
{
    if (p_material.m_base_color_sampler_feedback_uav != RAL_NULL_BINDLESS_INDEX ||
        p_material.m_metallic_roughness_sampler_feedback_uav != RAL_NULL_BINDLESS_INDEX ||
        p_material.m_normal_map_sampler_feedback_uav != RAL_NULL_BINDLESS_INDEX)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//-----------------------------------------------------------------------------
pbr_material_t get_material(uint p_hierarchy_srv,
                            uint p_geo_index,
                            raytracing_mesh_vertex_t p_mesh_vertex,
                            uint p_path_length)
{
#if defined(DEBUG_NO_MATERIALS)
    {
        pbr_material_t l_material;

        l_material.m_base_color             = 1.0f;
        l_material.m_diffuse_reflectance    = 1.0f;
        l_material.m_specular_f0            = 0;
        l_material.m_emissive               = 0;
        l_material.m_roughness              = 1.0f;
        l_material.m_metallic               = 0;
        l_material.m_ior                    = 1.0f;
        l_material.m_normal_ts              = float3(0, 0, 1.0f);
        l_material.m_opacity                = 0;
        l_material.m_alpha_cutoff           = 0;

        return l_material;
    }
#endif

    // Get hierarchy
    StructuredBuffer<sb_raytracing_hierarchy_t> l_hierarchy_buffer = ResourceDescriptorHeap[NonUniformResourceIndex(p_hierarchy_srv)];
    sb_raytracing_hierarchy_t l_hierarchy = l_hierarchy_buffer[p_geo_index];
    //sb_render_item_t l_render_item = l_hierarchy.m_render_item;

    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_hierarchy.m_render_item_id];

    // Early out if no material is assigned
    if (l_render_item.m_material_index == RAL_NULL_BINDLESS_INDEX)
    {
        return (pbr_material_t)0;
    }

    // Get material
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[l_render_item.m_material_index];

    pbr_material_t l_material;

    // Emission texture
    // TODO: just use NonUniformResourceIndex(...) in-place. Remove bindless_tex2d_sample_level_nonuniform
    float4 emission_texture = bindless_tex2d_sample_level(  NonUniformResourceIndex(l_pbr_material.m_emission_texture_srv),
                                                            (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                            p_mesh_vertex.m_uv0,
                                                            MIP_LEVEL,
                                                            1.0f);

    // Emission
    l_material.m_emissive = l_pbr_material.m_emissive_factor * emission_texture.rgb;

    // TODO: this is temporary for the integration
    uint2 l_screen_xy = 0;

    // Albedo
    float4 l_base_color_texture = 0;
    if (l_pbr_material.m_base_color_residency_buffer_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_base_color_texture = raytracing_bindless_tex2d_sample_level_with_feedback(NonUniformResourceIndex(l_pbr_material.m_base_color_texture_srv),
                                                                                    NonUniformResourceIndex(l_pbr_material.m_base_color_residency_buffer_srv),
                                                                                    l_pbr_material.m_base_color_residency_buffer_dim,
                                                                                    (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                                                    p_mesh_vertex.m_uv0,
                                                                                    MIP_LEVEL,
                                                                                    1.0);
    }
    else
    {
        l_base_color_texture = bindless_tex2d_sample_level( NonUniformResourceIndex(l_pbr_material.m_base_color_texture_srv),
                                                            (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                            p_mesh_vertex.m_uv0,
                                                            MIP_LEVEL,
                                                            1.0f);
    }

    l_material.m_base_color = l_base_color_texture.rgb;
    l_material.m_base_color = gamma_to_linear(l_pbr_material.m_base_color_factor.xyz * l_material.m_base_color);

    // Metallic-roughness
    float2 l_metallic_roughness = 0;
    if(l_pbr_material.m_metallic_roughness_residency_buffer_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_metallic_roughness = raytracing_bindless_tex2d_sample_level_with_feedback(NonUniformResourceIndex(l_pbr_material.m_metallic_roughness_texture_srv),
                                                                                    NonUniformResourceIndex(l_pbr_material.m_metallic_roughness_residency_buffer_srv),
                                                                                    l_pbr_material.m_metallic_roughness_residency_buffer_dim,
                                                                                    (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                                                    p_mesh_vertex.m_uv0,
                                                                                    MIP_LEVEL,
                                                                                    1.0).bg;
    }else
    {
        l_metallic_roughness = bindless_tex2d_sample_level( NonUniformResourceIndex(l_pbr_material.m_metallic_roughness_texture_srv),
                                                            (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                            p_mesh_vertex.m_uv0,
                                                            MIP_LEVEL,
                                                            1.0f).bg;
    }

    l_material.m_metallic = l_pbr_material.m_metallic_factor * l_metallic_roughness.x;
    l_material.m_roughness = l_pbr_material.m_roughness_factor * l_metallic_roughness.y;
    l_material.m_roughness = l_material.m_roughness * l_material.m_roughness; // TODO: add function for converting to perceptual roughness

    // Normal
    float2 l_normal_texture = 0;
    if(l_pbr_material.m_normal_map_residency_buffer_srv != RAL_NULL_BINDLESS_INDEX)
    {
        l_normal_texture = raytracing_bindless_tex2d_sample_level_with_feedback(NonUniformResourceIndex(l_pbr_material.m_normal_map_texture_srv),
                                                                                NonUniformResourceIndex(l_pbr_material.m_normal_map_residency_buffer_srv),
                                                                                l_pbr_material.m_normal_map_residency_buffer_dim,
                                                                                (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                                                p_mesh_vertex.m_uv0,
                                                                                MIP_LEVEL,
                                                                                float4(0.5f, 0.5f, 1.0, 0)).rg;
    }else
    {
        l_normal_texture = bindless_tex2d_sample_level( NonUniformResourceIndex(l_pbr_material.m_normal_map_texture_srv),
                                                        (SamplerState) SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP],
                                                        p_mesh_vertex.m_uv0,
                                                        MIP_LEVEL,
                                                        float4(0.5f, 0.5f, 1.0f, 0)).xy;
    }

    l_normal_texture = l_normal_texture * 2.0f - 1.0f;
    l_material.m_normal_ts.xy = l_normal_texture.xy;
    l_material.m_normal_ts.z = sqrt(1.0f - saturate(dot(l_normal_texture.xy, l_normal_texture.xy)));

    // Index of refraction(IOR)
    l_material.m_ior = 1.0f;


    l_material.m_opacity        = l_pbr_material.m_base_color_factor.w * l_base_color_texture.w;
    l_material.m_alpha_cutoff   = l_pbr_material.m_alpha_cutoff;

    // PBR deriviations
    // Diffuse color
    l_material.m_diffuse_reflectance    = base_color_to_diffuse_reflectance(l_material.m_base_color, l_material.m_metallic);
    l_material.m_specular_f0            = base_color_to_specular_f0(l_material.m_base_color, l_material.m_metallic);

#if !defined(MB_RAYTRACING_NO_VT_FEEDBACK)
    // Write data to do sample feedback later
    if (p_path_length <= 1 &&
        g_push_constants.m_vt_feedback_buffer_capacity != 0 &&
        is_material_vt(l_pbr_material) > 0)
    {
        RWStructuredBuffer<uint> l_vt_feedback_buffer_counter = ResourceDescriptorHeap[RAYTRACING_FEEDBACK_BUFFER_COUNTER_UAV];
        RWStructuredBuffer<sb_vt_feedback_t> l_vt_feedback_buffer = ResourceDescriptorHeap[RAYTRACING_FEEDBACK_BUFFER_UAV];

        // Get index
        uint l_index = 0;
        InterlockedAdd(l_vt_feedback_buffer_counter[0], 1, l_index);

        // Fill the feedback
        if (l_index < g_push_constants.m_vt_feedback_buffer_capacity)
        {
            l_vt_feedback_buffer[l_index].m_render_item_id = l_hierarchy.m_render_item_id;
            l_vt_feedback_buffer[l_index].m_uv = p_mesh_vertex.m_uv0;
            l_vt_feedback_buffer[l_index].m_min_mip_level = MIP_LEVEL;
        }
    }
#endif
    return l_material;
}

//-----------------------------------------------------------------------------
float3 apply_scattering(float3 p_ray_start,
                        float3 p_ray_dir,
                        float p_ray_length,
                        float3 p_incoming_radiance,
                        cb_camera_t p_camera,
                        float p_sund_disk_enabled)
{
    StructuredBuffer<cb_push_atmospheric_scattering_t> l_atmospheric_scattering_buffer = ResourceDescriptorHeap[g_push_constants.m_atmospheric_scattering_srv];
    cb_push_atmospheric_scattering_t l_atmospheric_scattering_settings = l_atmospheric_scattering_buffer[0];

    // Calculate inscattering
    float3 l_light_inscattering = 0;
    float3 l_light_extinction = 0;
    compute_inscattering_along_ray( p_ray_start,
                                    p_ray_dir,
                                    p_ray_length,
                                    l_atmospheric_scattering_settings,
                                    l_light_inscattering,
                                    l_light_extinction);

    // Atmospheric scattering
    float3 l_radiance = p_incoming_radiance * l_light_extinction + max(0, l_light_inscattering);

    // Add sun disk
    float3 l_sun_disk = get_sun_disc_mask(l_atmospheric_scattering_settings.m_sun_light_dir, p_ray_dir);
    l_radiance += (p_sund_disk_enabled == SKY_DISK_ENABLED) * l_sun_disk * l_atmospheric_scattering_settings.m_sun_light_color * l_light_extinction;

    return l_radiance;
}

//-----------------------------------------------------------------------------
void get_inscattering_and_extinction(   float3 p_ray_start,
                                        float3 p_ray_dir,
                                        float p_ray_length,
                                        cb_camera_t p_camera,
                                        float p_sund_disk_enabled,
                                        out float3 p_inscattering,
                                        out float3 p_extinction)
{
    StructuredBuffer<cb_push_atmospheric_scattering_t> l_atmospheric_scattering_buffer = ResourceDescriptorHeap[g_push_constants.m_atmospheric_scattering_srv];
    cb_push_atmospheric_scattering_t l_atmospheric_scattering_settings = l_atmospheric_scattering_buffer[0];

    // Calculate inscattering
    compute_inscattering_along_ray( p_ray_start,
                                    p_ray_dir,
                                    p_ray_length,
                                    l_atmospheric_scattering_settings,
                                    p_inscattering,
                                    p_extinction);

    // Add sun disk
    float3 l_sun_disk = get_sun_disc_mask(l_atmospheric_scattering_settings.m_sun_light_dir, p_ray_dir);
    p_inscattering += (p_sund_disk_enabled == SKY_DISK_ENABLED) * l_sun_disk * l_atmospheric_scattering_settings.m_sun_light_color * p_extinction;
}

//-----------------------------------------------------------------------------
void shade_point(   inout pl_ray_payload_t p_payload,
                    raytracing_mesh_vertex_t p_mesh_vertex,
                    pbr_material_t p_material)
{
    //-------------------------------------------------------------------------
    // TBN
    //-------------------------------------------------------------------------

    float3x3 l_tbn = float3x3(normalize(p_mesh_vertex.m_tangent), normalize(p_mesh_vertex.m_binormal), normalize(p_mesh_vertex.m_normal));

#if 1
    // Transform normal into world space
    float3 l_normal_ws = normalize(mul(p_material.m_normal_ts, l_tbn));

    // Update TBN
    l_tbn._31_32_33 = l_normal_ws;

    // Re-construct TBN
    //float3 l_tangent = abs(dot(float3(1.0, 0, 0), l_normal_ws)) > 0.99 ? float3(0, 0, 1.0) : float3(1.0, 0, 0);
    //float3 l_binormal = normalize(cross(float3(1.0, 0, 0), l_normal_ws));
    //l_tangent = normalize(cross(l_normal_ws, l_binormal));
    //l_tbn = float3x3(l_tangent, l_binormal, normalize(l_normal_ws));
#else
    float3 l_normal_ws = normalize(p_mesh_vertex.m_normal);
#endif

    //-------------------------------------------------------------------------
    // Ray params
    //-------------------------------------------------------------------------

    float l_hit_t = RayTCurrent();
    float3 l_incoming_ray_dir_ws = WorldRayDirection();
    float3 l_ray_origin_ws = WorldRayOrigin();

    // Find the world-space hit position
    float3 l_pos_w = l_ray_origin_ws + l_hit_t * l_incoming_ray_dir_ws;


    //-------------------------------------------------------------------------
    // Emission
    //-------------------------------------------------------------------------

    p_payload.m_radiance = g_push_constants.m_emissive_lighting > 0 ? p_payload.m_throughput * p_material.m_emissive : 0;

    //-------------------------------------------------------------------------
    // Direct lighting
    //-------------------------------------------------------------------------

    // TODO: test section
    float3 l_view_dir = -l_incoming_ray_dir_ws;
    float3 l_diffuse_reflectance = p_material.m_diffuse_reflectance;
    float3 l_specular_f0 = p_material.m_specular_f0;
    float l_ggx_roughness = sqrt(p_material.m_roughness);
    float l_metallic = p_material.m_metallic;

    // Direct lighting
    float3 l_direct_lighting = 0;
    if (g_push_constants.m_direct_lighting > 0)
    {
        ConstantBuffer<cb_light_list_t> l_light_list = ResourceDescriptorHeap[g_push_constants.m_light_list_cbv];
        for (int l_i = 0; l_i < MB_MAX_LIGHTS; l_i++)
        {
            cb_light_t l_light = l_light_list.m_light_list[l_i];
            if (l_light.m_type == LIGHT_TYPE_DIRECTIONAL) // Directional light
            {
#if ENABLE_DIRECTIONAL_LIGHT
                directional_light_t l_directional_light = get_directional_light_param(l_light);
                float3 l_lighting = ggx_direct_lighting(l_normal_ws, l_view_dir, l_directional_light.m_direction,
                                                        l_diffuse_reflectance.xyz, l_specular_f0, l_ggx_roughness);

                l_direct_lighting += l_lighting * l_directional_light.m_color;

                // TODO:
                // Shadow ray
                if (g_push_constants.m_raytracing_acc_struct_srv != RAL_NULL_BINDLESS_INDEX &&
                    dot(l_normal_ws, l_directional_light.m_direction) > 0.00001f)
                {
                    RaytracingAccelerationStructure l_acc_struct = ResourceDescriptorHeap[g_push_constants.m_raytracing_acc_struct_srv];

                    // Shadow ray
                    RayDesc l_shadow_ray;
                    l_shadow_ray.Origin = offset_ray(l_pos_w, p_mesh_vertex.m_normal);
#if defined(SAMPLE_SUN_AS_DISK)
                    // Pick a sample on a sun disk
                    float l_r0 = random_float01(p_payload.m_rand_state);
                    float l_r1 = random_float01(p_payload.m_rand_state);
                    float2 l_disk_sample = sample_concentric_disk(float2(l_r0, l_r1));
                    l_disk_sample *= tan(SUN_ANGLE_RADIUS);

                    float3 l_view = l_directional_light.m_direction;
                    float3 l_t1 = (l_view.z < 0.999f) ? normalize(cross(l_view, float3(0, 0, 1))) : float3(1, 0, 0);
                    float3 l_t2 = cross(l_t1, l_view);

                    l_shadow_ray.Direction = l_directional_light.m_direction + l_t1 * l_disk_sample.x + l_t2 * l_disk_sample.y;
#else
                    l_shadow_ray.Direction = l_directional_light.m_direction;
#endif
                    l_shadow_ray.TMin = 0.01;
                    l_shadow_ray.TMax = MAX_SHADOW_RAY_DISTANCE;

                    // Trace rays
                    pl_shadow_ray_payload_t l_shadow_payload;
                    l_shadow_payload.m_hit = true; // Initialize to true, because we are ignoring closest hit shader execution

                    TraceRay(l_acc_struct, // AccelerationStructure
                                RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, // RayFlags
                                0xFFFFFFFF, // InstanceInclusionMask
                                SHADOW_RAY_INDEX, // RayContributionToHitGroupIndex
                                0, // MultiplierForGeometryContributionToHitGroupIndex
                                SHADOW_RAY_INDEX, // MissShaderIndex
                                l_shadow_ray, // Ray
                                l_shadow_payload); // Payload

                    // Get shadow factor
                    l_direct_lighting *= l_shadow_payload.m_hit ? 0.0 : 1.0;
                }
#endif
            }
            else if (l_light.m_type == LIGHT_TYPE_OMNI) // Omni-directional light
            {
#if 0 //ENABLE_OMNI_LIGHT
                omni_light_t l_omni_light = get_omni_light_param(l_light);
                float l_atten = omni_light_atten(p_input.m_position_ws_local, l_omni_light.m_position_ws_local, l_omni_light.m_range);
                float3 l_light_dir = normalize(l_omni_light.m_position_ws_local - p_input.m_position_ws_local);
                float3 l_lighting = ggx_direct_lighting(l_normal_ws, l_view_dir, l_light_dir, l_diffuse_reflectance.xyz,
                                                        l_specular_f0, l_ggx_roughness);

                l_direct_lighting += l_lighting * l_omni_light.m_color * l_atten;
#endif
            }
            else if (l_light.m_type == LIGHT_TYPE_SPOT) // Spot light
            {
#if 0 //ENABLE_SPOT_LIGHT
                spot_light_t l_spot_light = get_spot_light_param(l_light);
                float l_atten = spot_light_atten(p_input.m_position_ws_local, l_spot_light.m_position_ws_local, l_spot_light.m_range,
                                                 l_spot_light.m_direction, l_spot_light.m_angle_scale, l_spot_light.m_angle_offset);
                float3 l_light_dir = normalize(l_spot_light.m_position_ws_local - p_input.m_position_ws_local);
                float3 l_lighting = ggx_direct_lighting(l_normal_ws, l_view_dir, l_light_dir, l_diffuse_reflectance.xyz,
                                                        l_specular_f0, l_ggx_roughness);

                l_direct_lighting += l_lighting * l_spot_light.m_color * l_atten;
#endif
            }
            else // LIGHT_TYPE_NULL
            {
                break;
            }
        }
    }

    p_payload.m_radiance += p_payload.m_throughput * l_direct_lighting;

    //-------------------------------------------------------------------------
    // Russian roulette
    //-------------------------------------------------------------------------

    if (p_payload.m_path_length >= RUSSIAN_ROULETTE_MIN_BOUNCE)
    {
        float l_russian_roulette_prob = min(0.95f, get_luminance(p_payload.m_throughput));
        if (l_russian_roulette_prob < random_float01(p_payload.m_rand_state))
        {
            return;
        }
        else
        {
            p_payload.m_throughput /= l_russian_roulette_prob;
        }
    }

    //-------------------------------------------------------------------------
    // Next bounce
    //-------------------------------------------------------------------------

    if (p_payload.m_path_length < g_push_constants.m_num_bounces)
    {
        RaytracingAccelerationStructure l_acc_struct = ResourceDescriptorHeap[g_push_constants.m_raytracing_acc_struct_srv];

        //---------------------------------------------------------------------
        // Choose between specular and diffuse ray
        //---------------------------------------------------------------------

        float l_specular_probability = get_specular_brdf_probability(p_material.m_diffuse_reflectance,
                                                                     p_material.m_specular_f0,
                                                                     l_view_dir,
                                                                     l_normal_ws);

#if defined(MB_RAYTRACING_DIFFUSE_GI)
        bool l_use_diffuse_ray = true;
        bool l_use_specular_ray = false;
#else
        bool l_use_diffuse_ray = true;
        bool l_use_specular_ray = true;
        if (random_float01(p_payload.m_rand_state) < l_specular_probability)
        {
            l_use_diffuse_ray = false;
            l_use_specular_ray = true;
            p_payload.m_throughput /= l_specular_probability;
        }
        else
        {
            l_use_diffuse_ray = true;
            l_use_specular_ray = false;
            p_payload.m_throughput /= (1.0f - l_specular_probability);
        }
#endif

        //---------------------------------------------------------------------
        // Diffuse rays
        //---------------------------------------------------------------------

        float3 l_radiance = 0;

        if (g_push_constants.m_indirect_diffuse &&
            l_use_diffuse_ray)
        {
            for (uint l_ray_idx_0 = 0; l_ray_idx_0 < g_push_constants.m_num_samples; ++l_ray_idx_0)
            {
                float r0 = random_float01(p_payload.m_rand_state);
                float r1 = random_float01(p_payload.m_rand_state);
#if defined(USE_UNIFORM_SAMPLING)
                float3 l_ray_dir_ts = sample_hemisphere_uniform(r0, r1);
#else
                float3 l_ray_dir_ts = sample_hemisphere_cosine(r0, r1);
#endif

                float3 l_ray_dir_ws = normalize(mul(l_ray_dir_ts, l_tbn));
                float3 l_ray_dir = l_ray_dir_ws;
                //float3 l_throughput = p_payload.m_path_length > 0 ? 1.0 : p_material.m_diffuse_reflectance;

                float3 l_throughput = p_material.m_diffuse_reflectance;

                /*if (dot(l_ray_dir_ws, l_normal_ws) < 0)
                {
                    l_throughput = float3(1.0, 0, 0);
                }*/

#if defined(USE_UNIFORM_SAMPLING)
                // Diffuse D = (I * NdotL / Pi)
                // Unifrom hemisphere sampling PDF = 1.0 / (2.0 * PI)
                // According to Monte-Carlo integration D / PDF = 2.0 * I * NdotL
                // Dot is always positive, because l_ray_dir_ws is upper hemisphere

                l_throughput *= 2.0f * dot(l_ray_dir_ws, l_normal_ws);
#else
                // Diffuse D = (I * NdotL / Pi)
                // Cosine hemisphere sampling PDF = NdotL / PI
                // According to Monte-Carlo integration D / PDF = I
                l_throughput *= 1.0f;
#endif

                // Bounce ray
                RayDesc l_ray;
                l_ray.Origin = offset_ray(l_pos_w, p_mesh_vertex.m_normal);
                l_ray.Direction = l_ray_dir;
                l_ray.TMin = 0.00001;
                l_ray.TMax = MAX_STANDARD_RAY_DISTANCE;

                // Trace ray
                pl_ray_payload_t l_payload_tmp;
                l_payload_tmp.m_path_length = p_payload.m_path_length + 1;
                l_payload_tmp.m_rand_state = p_payload.m_rand_state;
                l_payload_tmp.m_throughput = l_throughput * p_payload.m_throughput;
                l_payload_tmp.m_radiance = 0;

                TraceRay(l_acc_struct, // AccelerationStructure
                            0, // RayFlags
                            0xFFFFFFFF, // InstanceInclusionMask
                            0, // RayContributionToHitGroupIndex
                            0, // MultiplierForGeometryContributionToHitGroupIndex
                            0, // MissShaderIndex
                            l_ray, // Ray
                            l_payload_tmp); // Payload

                // Continue updating seed.
                p_payload.m_rand_state = l_payload_tmp.m_rand_state;

                l_radiance += l_payload_tmp.m_radiance;
            }
        }

        //---------------------------------------------------------------------
        // Specular rays
        //---------------------------------------------------------------------

        if (g_push_constants.m_indirect_specular &&
            l_use_specular_ray)
        {
            for (uint l_ray_idx_1 = 0; l_ray_idx_1 < g_push_constants.m_num_samples; ++l_ray_idx_1)
            {
                float r0 = random_float01(p_payload.m_rand_state);
                float r1 = random_float01(p_payload.m_rand_state);

#define SPECULAR_IMPORTANCE_SAMPLING
#if defined(SPECULAR_IMPORTANCE_SAMPLING)
                float3 incomingRayDirTS = normalize(mul(l_incoming_ray_dir_ws, transpose(l_tbn)));
                float3 microfacetNormalTS = sample_ggx_visible_normal(-incomingRayDirTS, l_ggx_roughness * l_ggx_roughness, l_ggx_roughness * l_ggx_roughness, r0, r1);
                float3 sampleDirTS = reflect(incomingRayDirTS, microfacetNormalTS);
                float3 l_ray_dir_ts = sampleDirTS;
#else
#if defined(USE_UNIFORM_SAMPLING)
                float3 l_ray_dir_ts = sample_hemisphere_uniform(r0, r1);
#else
                float3 l_ray_dir_ts = sample_hemisphere_cosine(r0, r1);
#endif
#endif

                float3 l_ray_dir_ws = normalize(mul(l_ray_dir_ts, l_tbn));
                float3 l_ray_dir = l_ray_dir_ws;

#if defined(SPECULAR_IMPORTANCE_SAMPLING)
                float3 l_throughput = ggx_specular_lighting(l_normal_ws,
                                                            l_view_dir,
                                                            l_ray_dir_ws,
                                                            l_specular_f0,
                                                            l_ggx_roughness);

                float l_vdh = max(0.001f, min(1.0f, dot(-incomingRayDirTS, microfacetNormalTS)));
                float3 l_f = fresnel_reflectance_schlick(l_specular_f0, l_vdh);
                float l_g1 = smith_g1(l_normal_ws, l_ray_dir_ws, -l_incoming_ray_dir_ws, l_ggx_roughness * l_ggx_roughness * l_ggx_roughness * l_ggx_roughness);
                float l_g2 = smith_g2(l_normal_ws, l_ray_dir_ws, -l_incoming_ray_dir_ws, l_ggx_roughness * l_ggx_roughness * l_ggx_roughness * l_ggx_roughness);

                l_throughput = l_f * (l_g2 / l_g1);
#else
                float3 l_throughput = ggx_specular_lighting(l_normal_ws,
                                                            l_view_dir,
                                                            l_ray_dir_ws,
                                                            l_specular_f0,
                                                            l_ggx_roughness);

#if defined(USE_UNIFORM_SAMPLING)
                l_throughput *= 2.0f * M_PI;
#else
                l_throughput *= M_PI / dot(l_normal_ws, l_ray_dir_ws);
#endif
#endif

                // Bounce ray
                RayDesc l_ray;
                l_ray.Origin = offset_ray(l_pos_w, p_mesh_vertex.m_normal);
                l_ray.Direction = l_ray_dir;
                l_ray.TMin = 0.00001;
                l_ray.TMax = MAX_STANDARD_RAY_DISTANCE;

                // Trace ray
                pl_ray_payload_t l_payload_tmp;
                l_payload_tmp.m_path_length = p_payload.m_path_length + 1;
                l_payload_tmp.m_rand_state = p_payload.m_rand_state;
                l_payload_tmp.m_throughput = l_throughput * p_payload.m_throughput;
                l_payload_tmp.m_radiance = 0;

                TraceRay(l_acc_struct, // AccelerationStructure
                            0, // RayFlags
                            0xFFFFFFFF, // InstanceInclusionMask
                            0, // RayContributionToHitGroupIndex
                            0, // MultiplierForGeometryContributionToHitGroupIndex
                            0, // MissShaderIndex
                            l_ray, // Ray
                            l_payload_tmp); // Payload

                // Continue updating seed.
                p_payload.m_rand_state = l_payload_tmp.m_rand_state;

                l_radiance += l_payload_tmp.m_radiance;
            }
        }

        p_payload.m_radiance += l_radiance / g_push_constants.m_num_samples;

        //-------------------------------------------------------------------------
        // Scattering
        //-------------------------------------------------------------------------
#if defined(ATMOSPHERIC_SCATTERING_ON_HIT)
        if (p_payload.m_path_length == 1)
        {
            // Get camera data
            ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

            // Get initial attributes of the ray
            float3 l_ray_start = WorldRayOrigin() + l_camera.m_camera_pos;
            float3 l_ray_dir = WorldRayDirection();
            float l_ray_length = RayTCurrent();

            p_payload.m_radiance = apply_scattering(l_ray_start,
                                                    l_ray_dir,
                                                    l_ray_length,
                                                    p_payload.m_radiance,
                                                    l_camera,
                                                    SKY_DISK_DISABLED);
        }
#endif
    }
}

//-----------------------------------------------------------------------------
float3 min_diff(float3 p_position, float3 p_position_right, float3 p_position_left)
{
    float3 l_v1 = p_position_right - p_position;
    float3 l_v2 = p_position - p_position_left;
    return (dot(l_v1, l_v1) < dot(l_v2, l_v2)) ? l_v1 : l_v2;
}

//-----------------------------------------------------------------------------
float3 reconstruct_normal(cb_camera_t l_camera, float2 l_uv)
{
    // Get remapped uv
    float2 l_remapped_uv = get_remapped_uv(l_uv, l_camera.m_render_scale);

    float2 l_uv0 = l_remapped_uv;
    float2 l_uv1 = l_remapped_uv + float2( g_push_constants.m_inv_dst_resolution.x, 0); // right
    float2 l_uv2 = l_remapped_uv + float2(0,  g_push_constants.m_inv_dst_resolution.y); // top
    float2 l_uv3 = l_remapped_uv + float2(-g_push_constants.m_inv_dst_resolution.x, 0); // left
    float2 l_uv4 = l_remapped_uv + float2(0, -g_push_constants.m_inv_dst_resolution.y); // bottom

    // Get depth
    float l_depth0 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv0).r;
    float l_depth1 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv1).r;
    float l_depth2 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv2).r;
    float l_depth3 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv3).r;
    float l_depth4 = bindless_tex2d_sample_level(g_push_constants.m_depth_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_POINT_CLAMP], l_uv4).r;

    // Discard samples if they fall outside of the screen
    l_depth1 *= l_uv1.x > 1.0f ? 0 : 1.0f;
    l_depth2 *= l_uv2.y > 1.0f ? 0 : 1.0f;
    l_depth3 *= l_uv3.x < 0.0f ? 0 : 1.0f;
    l_depth4 *= l_uv4.y < 0.0f ? 0 : 1.0f;

    // Get world space local position
    float3 l_p0 = get_world_space_local_position(l_uv0, l_depth0, l_camera.m_inv_view_proj_local);
    float3 l_p1 = get_world_space_local_position(l_uv1, l_depth1, l_camera.m_inv_view_proj_local);
    float3 l_p2 = get_world_space_local_position(l_uv2, l_depth2, l_camera.m_inv_view_proj_local);
    float3 l_p3 = get_world_space_local_position(l_uv3, l_depth3, l_camera.m_inv_view_proj_local);
    float3 l_p4 = get_world_space_local_position(l_uv4, l_depth4, l_camera.m_inv_view_proj_local);

    float3 l_normal = normalize(cross(min_diff(l_p0, l_p1, l_p3), min_diff(l_p0, l_p2, l_p4)));
    return l_normal;
}

//-----------------------------------------------------------------------------
downsampled_data_t get_max_depth(uint2 p_launch_index)
{
    Texture2D<float> l_depth_buffer = ResourceDescriptorHeap[g_push_constants.m_depth_texture_srv];
    Texture2D<float2> l_velocity_srv = ResourceDescriptorHeap[g_push_constants.m_velocity_texture_srv];

    uint2 l_low_res_coords = g_push_constants.m_raytracing_resolution_scale * p_launch_index.xy;

    // Initialize struct
    downsampled_data_t l_downsampled_data = (downsampled_data_t)0;
    l_downsampled_data.m_depth = -FLT_MAX;
    l_downsampled_data.m_velocity = 0;

    // Make sure all attributes are selected in-sync with depth
    // Same pixel coords selected for depth must be used to sample other data
    uint2 l_sample_location = l_low_res_coords;
    for (uint l_x = 0; l_x < g_push_constants.m_raytracing_resolution_scale; ++l_x)
    {
        for (uint l_y = 0; l_y < g_push_constants.m_raytracing_resolution_scale; ++l_y)
        {
            uint2 l_high_res_coord = l_low_res_coords + uint2(l_x, l_y);
            float l_depth = l_depth_buffer.Load(uint3(l_high_res_coord, 0));

            // MAX filter
            if (l_downsampled_data.m_depth < l_depth)
            {
                l_sample_location = l_high_res_coord;
                l_downsampled_data.m_depth = l_depth;
            }
        }
    }

    l_downsampled_data.m_velocity           = l_velocity_srv[l_sample_location.xy];
    l_downsampled_data.m_offset_in_pixels   = l_sample_location.xy - l_low_res_coords;

    return l_downsampled_data;
}

//-----------------------------------------------------------------------------
downsampled_data_t get_min_depth(uint2 p_launch_index)
{
    Texture2D<float> l_depth_buffer = ResourceDescriptorHeap[g_push_constants.m_depth_texture_srv];
    Texture2D<float2> l_velocity_srv = ResourceDescriptorHeap[g_push_constants.m_velocity_texture_srv];

    uint2 l_low_res_coords = g_push_constants.m_raytracing_resolution_scale * p_launch_index.xy;

    // Initialize struct
    downsampled_data_t l_downsampled_data = (downsampled_data_t)0;
    l_downsampled_data.m_depth = FLT_MAX;
    l_downsampled_data.m_velocity = 0;

    // Make sure all attributes are selected in-sync with depth
    // Same pixel coords selected for depth must be used to sample other data
    uint2 l_sample_location = l_low_res_coords;
    for (uint l_x = 0; l_x < g_push_constants.m_raytracing_resolution_scale; ++l_x)
    {
        for (uint l_y = 0; l_y < g_push_constants.m_raytracing_resolution_scale; ++l_y)
        {
            uint2 l_high_res_coord = l_low_res_coords + uint2(l_x, l_y);
            float l_depth = l_depth_buffer.Load(uint3(l_high_res_coord, 0));

            // MIN filter
            if (l_downsampled_data.m_depth > l_depth)
            {
                l_sample_location = l_high_res_coord;
                l_downsampled_data.m_depth = l_depth;
            }
        }
    }

    l_downsampled_data.m_velocity           = l_velocity_srv[l_sample_location.xy];
    l_downsampled_data.m_offset_in_pixels   = l_sample_location.xy - l_low_res_coords;

    return l_downsampled_data;
}

//-----------------------------------------------------------------------------
// Ray generation
//-----------------------------------------------------------------------------

[shader("raygeneration")]
void raygen_shader()
{
    uint3 l_launch_index = DispatchRaysIndex();
    uint3 l_launch_dim = DispatchRaysDimensions();

    // UAVs
    RWTexture2D<float4> l_accumulation_buffer_write = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_uav];

    // Initialize the payload
    pl_ray_payload_t l_payload = (pl_ray_payload_t)0;
    l_payload.m_path_length = 1;
    l_payload.m_throughput  = 1.0f;
    l_payload.m_radiance    = 0;

    // Generate new frame only when needed
    if(g_push_constants.m_generate_new_frame)
    {
        RaytracingAccelerationStructure l_acc_struct = ResourceDescriptorHeap[g_push_constants.m_raytracing_acc_struct_srv];
        ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

        // Init random generator
        rand_type_t l_rand_state = init_rand(l_launch_index.xy, l_launch_dim.xy, g_push_constants.m_acc_frame_index);

        // calculate subpixel camera jitter for anti aliasing
#if defined(CAMERA_SUPERSAMPLING)
        float2 l_jitter = g_push_constants.m_acc_frame_index == 0 ? 0 : float2(random_float01(l_rand_state), random_float01(l_rand_state));
#else
        float2 l_jitter = 0;
#endif

        float2 l_pixel_pos = l_launch_index.xy + l_jitter;
        float2 l_pos_ndc = l_pixel_pos / ((float2)l_launch_dim.xy * 0.5f) - 1.0f;
        l_pos_ndc.y *= -1.0f;

        // Reproject screen coords into world direction
        float4 l_ray_end_tmp =  mul(float4(l_pos_ndc, Z_FAR, 1.0f), l_camera.m_inv_view_proj_local);
        l_ray_end_tmp.xyz /= l_ray_end_tmp.w;

        // Get ray params
        float3 l_ray_start  = g_push_constants.m_world_origin_local;
        float3 l_ray_dir    = normalize(l_ray_end_tmp.xyz);

        // Ray dsc
        RayDesc ray;
        ray.Origin      = l_ray_start;
        ray.Direction   = l_ray_dir;
        ray.TMin        = 0;
        ray.TMax        = MAX_STANDARD_RAY_DISTANCE;

        // Trace ray
#if defined(USE_HAMMERSLEY)
        l_payload.m_rand_state    = g_push_constants.m_acc_frame_index;
#else
        l_payload.m_rand_state    = l_rand_state;
#endif
        TraceRay(   l_acc_struct,       // AccelerationStructure
                    0,                  // RayFlags
                    0xFFFFFFFF,         // InstanceInclusionMask
                    STANDARD_RAY_INDEX, // RayContributionToHitGroupIndex
                    0,                  // MultiplierForGeometryContributionToHitGroupIndex
                    STANDARD_RAY_INDEX, // MissShaderIndex
                    ray,                // Ray
                    l_payload);         // Payload
    }

    // Get radiance from previous frame
    float3 l_previous_radiance = l_accumulation_buffer_write[l_launch_index.xy].xyz;
    if( g_push_constants.m_generate_new_frame &&
        g_push_constants.m_acc_frame_index == 0)
    {
        l_previous_radiance = 0;
    }

    // Radiance from a current frame
    float3 l_radiance = l_payload.m_radiance;

    // Store accumulated radiance into the history buffer
    float3 l_accumulated_radiance = l_previous_radiance + l_radiance;
    l_accumulation_buffer_write[l_launch_index.xy] = float4(l_accumulated_radiance, 1.0f);
}

//-----------------------------------------------------------------------------
[shader("raygeneration")]
void raygen_shader_diffuse_gi()
{
    uint3 l_launch_index = DispatchRaysIndex();
    uint3 l_launch_dim = DispatchRaysDimensions();

    // UAVs
    RWTexture2D<float4> l_accumulation_buffer_write = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_uav];
    RWTexture2D<float> l_frame_count_texture_write  = ResourceDescriptorHeap[g_push_constants.m_raytracing_frame_count_texture_uav];
    RWTexture2D<float3> l_normals_texture           = ResourceDescriptorHeap[g_push_constants.m_raytracing_normals_uav];

    // SRVs
    Texture2D l_accumulation_buffer_read            = ResourceDescriptorHeap[g_push_constants.m_raytracing_accumulation_rt_srv];
    Texture2D<float> l_frame_count_texture_read     = ResourceDescriptorHeap[g_push_constants.m_raytracing_frame_count_texture_srv];

    // Initialize the payload
    pl_ray_payload_t l_payload = (pl_ray_payload_t)0;
    l_payload.m_path_length = 1;
    l_payload.m_throughput  = 1.0f;
    l_payload.m_radiance    = 0;

    RaytracingAccelerationStructure l_acc_struct = ResourceDescriptorHeap[g_push_constants.m_raytracing_acc_struct_srv];
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get downsampled data
    downsampled_data_t l_downsampled_data = get_min_depth(l_launch_index.xy);

    // Sky is handled in atmospheric scattering
    // We don't need to output anything
    if (l_downsampled_data.m_depth == Z_FAR)
    {
        return;
    }

    uint2 l_full_res = l_launch_dim.xy * g_push_constants.m_raytracing_resolution_scale;
    uint2 l_low_res = l_launch_dim.xy;

    float2 l_uv_low_res = (float2)(l_launch_index.xy + 0.5f) / (float2)l_low_res;
    float2 l_uv_full_res = (float2) (g_push_constants.m_raytracing_resolution_scale * l_launch_index.xy + l_downsampled_data.m_offset_in_pixels + 0.5f) / (float2)l_full_res;

    // Get world space local position
    float3 l_pos_ws_local = get_world_space_local_position(l_uv_full_res, l_downsampled_data.m_depth, l_camera.m_inv_view_proj_local);

    // Reconstruct normal
    float3 l_normal_world_space = reconstruct_normal(l_camera, l_uv_full_res);

    {
        // Init random generator
        rand_type_t l_rand_state = init_rand(l_launch_index.xy, l_launch_dim.xy, g_push_constants.m_acc_frame_index %  MB_DIFFUSE_GI_NUM_JITTERED_POSITIONS);

        float3 l_ray_dir_ws = 0;

        {
            float r0 = random_float01(l_rand_state);
            float r1 = random_float01(l_rand_state);
#if defined(USE_UNIFORM_SAMPLING)
            float3 l_ray_dir_ts = sample_hemisphere_uniform(r0, r1);
#else
            float3 l_ray_dir_ts = sample_hemisphere_cosine(r0, r1);
#endif
            float3 up = abs(l_normal_world_space.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            float3 tangentX = normalize(cross(up, l_normal_world_space));
            float3 tangentY = cross(l_normal_world_space, tangentX);

            l_ray_dir_ws = normalize(tangentX * l_ray_dir_ts.x + tangentY * l_ray_dir_ts.y + l_normal_world_space * l_ray_dir_ts.z);
        }

        // Get ray params
        float3 l_ray_start = offset_ray(l_pos_ws_local + g_push_constants.m_world_origin_local, l_normal_world_space);

        // TODO:
        // Due to wind we have a miss-match in depth
        // This is a hack and should be solved properly
        l_ray_start += 0.1f * l_normal_world_space;

        //float3 l_ray_dir = reflect(l_ray_end_tmp.xyz, l_normal_world_space);
        float3 l_ray_dir = l_ray_dir_ws;
        
        // Ray dsc
        RayDesc l_ray;
        l_ray.Origin      = l_ray_start;
        l_ray.Direction   = l_ray_dir;
        l_ray.TMin        = 0;
        l_ray.TMax        = MAX_STANDARD_RAY_DISTANCE;

        // Trace ray
#if defined(USE_HAMMERSLEY)
        l_payload.m_rand_state    = g_push_constants.m_acc_frame_index;
#else
        l_payload.m_rand_state    = l_rand_state;
#endif
        TraceRay(   l_acc_struct,       // AccelerationStructure
                    0,                  // RayFlags
                    0xFFFFFFFF,         // InstanceInclusionMask
                    STANDARD_RAY_INDEX, // RayContributionToHitGroupIndex
                    0,                  // MultiplierForGeometryContributionToHitGroupIndex
                    STANDARD_RAY_INDEX, // MissShaderIndex
                    l_ray,              // Ray
                    l_payload);         // Payload

    }

    //--------------------------------------------------------------------------
    // Get pixel's velocity

    // Get world space local position
    float4 l_proj_pos = mul(float4(l_pos_ws_local, 1.0f), l_camera.m_view_proj_local_prev);
    float2 l_uv_prev_full_res = (l_proj_pos.xy / l_proj_pos.w) * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);

    // Convert to UV-space
    float2 l_object_velocity = l_downsampled_data.m_velocity * float2(0.5f, -0.5f);

    // Compute camera velocity
    float2 l_camera_velocity = (l_uv_full_res - l_uv_prev_full_res);

    // Add camera and object velocities
    float2 l_velocity = l_object_velocity + l_camera_velocity;

    //--------------------------------------------------------------------------
    // TAA

    float2 l_prev_coords_float = l_launch_index.xy - l_velocity * l_launch_dim.xy + 0.5f;
    int2 l_prev_coords_int = l_prev_coords_float;
    float2 l_prev_frame_uv = (float2)(l_prev_coords_float) / l_launch_dim.xy / (float)g_push_constants.m_raytracing_resolution_scale;

    float4 l_previous_radiance_depth    = l_accumulation_buffer_read.SampleLevel((SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_prev_frame_uv, 0);
    float3 l_previous_radiance          = l_previous_radiance_depth.xyz;
    float l_prev_depth                  = l_previous_radiance_depth.w;

    bool l_outside_of_frame = any(l_prev_coords_float < 0.45f) || any(l_prev_coords_float > l_launch_dim.xy - 0.45f);
    bool l_occlusion = l_outside_of_frame;

    // TODO: used for testing to always discard frame
#if 0
    l_occlusion = true;
    l_outside_of_frame = true;
#endif

    float l_conv_speed = l_outside_of_frame ? 0 : 0.95f;

    // Reject pixels based on reprojected depth comparison
#if 0
    l_conv_speed = abs(l_downsampled_data.m_depth - l_prev_depth) > 0.001f ? 0 : l_conv_speed;
#endif
#if 0
    l_occlusion = abs(l_downsampled_data.m_depth - l_prev_depth) > 0.001f ? true : l_occlusion;
#endif

    // Count the number of frame accumulated in the current pixel
    const float c_max_frame_count = 32.0f;
    float l_old_count = 255.0f * l_frame_count_texture_read.Load(uint3(l_prev_coords_int.xy, 0));
    float l_new_count = min(l_old_count + 1.0f, c_max_frame_count);
    l_new_count = l_occlusion ? 0 : l_new_count;
    l_frame_count_texture_write[l_launch_index.xy] = saturate(l_new_count / 255.0f);

    // Different frame accumulation techniques
#if 0
    float3 l_accumulated_radiance = lerp(l_payload.m_radiance, l_previous_radiance, l_conv_speed);
#else
    float3 l_accumulated_radiance = lerp(l_previous_radiance, l_payload.m_radiance, 1.0f / (1.0f + l_new_count));
#endif

    // Store accumulated radiance into the history buffer
    // Store depth used for generating this pixel into alpha channel. This will be used during upsampling
    l_accumulation_buffer_write[l_launch_index.xy] = float4(l_accumulated_radiance, l_downsampled_data.m_depth);

    // Store normals
    l_normals_texture[l_launch_index.xy] = l_normal_world_space;
}

//-----------------------------------------------------------------------------
// Primary rays
//-----------------------------------------------------------------------------

[shader("miss")]
void miss_shader(inout pl_ray_payload_t p_payload)
{
    if (g_push_constants.m_environment_lighting == 0)
    {
        p_payload.m_radiance = 0;
        return;
    }

    // Get camera data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Get tracing direction
    float3 l_ray_dir_ws = WorldRayDirection();
    l_ray_dir_ws = mul(l_ray_dir_ws, (float3x3)l_camera.m_align_ground_rotation);

    // Fetch cubemap
    int l_mip_offset = pow(8.0f, max(0, p_payload.m_path_length - 1));
    float3 l_color = bindless_texcube_sample_level(g_push_constants.m_cubemap_texture_srv, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP], l_ray_dir_ws, MIP_LEVEL + l_mip_offset).rgb;

    // Get exposure
    float l_multiplier = ev100_to_luminance(g_push_constants.m_exposure_value);

#if !defined(ATMOSPHERIC_SCATTERING_ON_MISS)
#if DEBUG_PROCESSING
    p_payload.m_radiance = p_payload.m_throughput * 3.0f;
#else
    p_payload.m_radiance = p_payload.m_throughput * l_multiplier * l_color;//float3(0.0, 0.0, 0.2);
#endif
#else
    // Get initial attributes of the ray
    float3 l_ray_start = WorldRayOrigin() + l_camera.m_camera_pos;
    float3 l_ray_dir = WorldRayDirection();
    float l_ray_length = 1000000000.0;//RayTCurrent(); //! \todo Put big number as we missed

    p_payload.m_radiance = p_payload.m_throughput * apply_scattering(   l_ray_start,
                                                                        l_ray_dir,
                                                                        l_ray_length,
                                                                        0,
                                                                        l_camera,
                                                                        p_payload.m_path_length == 1 ? SKY_DISK_ENABLED : SKY_DISK_DISABLED);
#endif
}

[shader("anyhit")]
void any_hit(inout pl_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    // System values
    uint l_instance_index = InstanceIndex();
    uint l_instance_id = InstanceID();
    uint l_primitive_index = PrimitiveIndex();
    uint l_geometry_index = GeometryIndex();
    float3x3 l_object_to_world_4x3 = (float3x3)ObjectToWorld4x3();

    // Unpack attributes
    float3 l_barycentrics = float3(1.0 - p_attribs.barycentrics.x - p_attribs.barycentrics.y, p_attribs.barycentrics.x, p_attribs.barycentrics.y);

    // Get hierarchy
    StructuredBuffer<sb_raytracing_hierarchy_t> l_hierarchy_buffer = ResourceDescriptorHeap[l_instance_id];
    sb_raytracing_hierarchy_t l_hierarchy = l_hierarchy_buffer[l_geometry_index];

    // Get buffer info
    raytracing_mesh_vertex_t l_mesh_vertex = get_vertex_mesh(l_hierarchy, l_geometry_index, l_primitive_index, l_barycentrics);

    // Get material
    pbr_material_t l_material = get_material(l_instance_id, l_geometry_index, l_mesh_vertex, p_payload.m_path_length);

    // At any hit, we test opacity and discard the hit if it's transparent
    if(l_material.m_opacity < l_material.m_alpha_cutoff)
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void closest_hit_shader(inout pl_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    // System values
    uint l_instance_index = InstanceIndex();
    uint l_instance_id  = InstanceID();
    uint l_primitive_index = PrimitiveIndex();
    uint l_geometry_index = GeometryIndex();
    float3x3 l_object_to_world_4x3 = (float3x3)ObjectToWorld4x3();

    // Unpack attributes
    float3 l_barycentrics = float3(1.0 - p_attribs.barycentrics.x - p_attribs.barycentrics.y, p_attribs.barycentrics.x, p_attribs.barycentrics.y);

    // Get hierarchy
    StructuredBuffer<sb_raytracing_hierarchy_t> l_hierarchy_buffer = ResourceDescriptorHeap[l_instance_id];
    sb_raytracing_hierarchy_t l_hierarchy = l_hierarchy_buffer[l_geometry_index];

    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_hierarchy.m_render_item_id];

    // Get buffer info
    raytracing_mesh_vertex_t l_mesh_vertex = get_vertex_mesh(l_hierarchy, l_geometry_index, l_primitive_index, l_barycentrics);

    // Get material
    pbr_material_t l_material = get_material(l_instance_id, l_geometry_index, l_mesh_vertex, p_payload.m_path_length);

    // Shade point
    shade_point(p_payload, l_mesh_vertex, l_material);
}

//-----------------------------------------------------------------------------
// Virtual Texture feedback loop
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Shadow rays
//-----------------------------------------------------------------------------

[shader("miss")]
void shadow_miss_shader(inout pl_shadow_ray_payload_t p_payload)
{
    p_payload.m_hit = false;
}

[shader("anyhit")]
void shadow_any_hit(inout pl_shadow_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    // System values
    uint l_instance_index = InstanceIndex();
    uint l_instance_id = InstanceID();
    uint l_primitive_index = PrimitiveIndex();
    uint l_geometry_index = GeometryIndex();
    float3x3 l_object_to_world_4x3 = (float3x3)ObjectToWorld4x3();

    // Unpack attributes
    float3 l_barycentrics = float3(1.0 - p_attribs.barycentrics.x - p_attribs.barycentrics.y, p_attribs.barycentrics.x, p_attribs.barycentrics.y);

    // Get hierarchy
    StructuredBuffer<sb_raytracing_hierarchy_t> l_hierarchy_buffer = ResourceDescriptorHeap[l_instance_id];
    sb_raytracing_hierarchy_t l_hierarchy = l_hierarchy_buffer[l_geometry_index];

    // Get buffer info
    raytracing_mesh_vertex_t l_mesh_vertex = get_vertex_mesh(l_hierarchy, l_geometry_index, l_primitive_index, l_barycentrics);

    // Get material
    pbr_material_t l_material = get_material(l_instance_id, l_geometry_index, l_mesh_vertex, NO_VT_FEEDBACK);

    // At any hit, we test opacity and discard the hit if it's transparent
    if (l_material.m_opacity < l_material.m_alpha_cutoff)
    {
        IgnoreHit();
    }
    else
    {
        AcceptHitAndEndSearch();
    }
}

[shader("closesthit")]
void shadow_closest_hit_shader(inout pl_shadow_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    p_payload.m_hit = true;
}

//-----------------------------------------------------------------------------
// Tile rays
//-----------------------------------------------------------------------------

[shader("closesthit")]
void tile_closest_shader_hit(inout pl_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    // System values
    uint l_instance_index = InstanceIndex();
    uint l_instance_id = InstanceID();
    uint l_primitive_index = PrimitiveIndex();
    uint l_geometry_index = GeometryIndex();
    float3x3 l_object_to_world_4x3 = (float3x3)ObjectToWorld4x3();

    // Unpack attributes
    float3 l_barycentrics = float3(1.0 - p_attribs.barycentrics.x - p_attribs.barycentrics.y, p_attribs.barycentrics.x, p_attribs.barycentrics.y);

    // Heirarchy SRV
    uint l_hierarchy_srv    = g_push_constants.m_tile_hierarchy_srv;
    uint l_tile_index       = l_instance_id;

    // Get hierarchy
    StructuredBuffer<sb_raytracing_tile_hierarchy_t> l_hierarchy_buffer = ResourceDescriptorHeap[l_hierarchy_srv];
    sb_raytracing_tile_hierarchy_t l_hierarchy = l_hierarchy_buffer[l_tile_index];
    sb_render_item_t l_render_item = l_hierarchy.m_render_item;

    // Get material
    sb_quadtree_material_t l_quadtree_material = (sb_quadtree_material_t) 0;
    if (l_render_item.m_material_buffer_srv != RAL_NULL_BINDLESS_INDEX)
    {
        StructuredBuffer<sb_quadtree_material_t> l_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
        l_quadtree_material = l_material_list[l_render_item.m_material_index];
    }

    // Get buffer info
    raytracing_mesh_vertex_t l_mesh_vertex = get_vertex_mesh_tile(l_render_item, l_primitive_index, l_barycentrics, l_quadtree_material);

    // Get tile material
    float2 l_tile_uv = tile_position_to_tile_uv(l_mesh_vertex.m_uv0);
    terrain_material_t l_terrain_material = sample_terrain_material(l_tile_index,
                                                                    l_tile_uv,
                                                                    l_quadtree_material.m_tile_height_array_index_srv,
                                                                    l_quadtree_material.m_tile_vt0_array_index_srv,
                                                                    l_quadtree_material.m_tile_vt1_array_index_srv,
                                                                    l_quadtree_material.m_elevation_tile_resolution,
                                                                    l_quadtree_material.m_elevation_tile_border,
                                                                    l_quadtree_material.m_vt_resolution,
                                                                    l_quadtree_material.m_vt_border);

    // Convert to PBR material
    pbr_material_t l_material;
    l_material.m_base_color             = l_terrain_material.m_base_color;
    l_material.m_emissive               = 0;
    l_material.m_roughness              = l_terrain_material.m_roughness;
    l_material.m_metallic               = 0;
    l_material.m_ior                    = 1.0f;
    l_material.m_normal_ts              = l_terrain_material.m_normal_ts;
    l_material.m_opacity                = 0;
    l_material.m_alpha_cutoff           = 0;
    l_material.m_diffuse_reflectance    = base_color_to_diffuse_reflectance(l_material.m_base_color, l_material.m_metallic);
    l_material.m_specular_f0            = base_color_to_specular_f0(l_material.m_base_color, l_material.m_metallic);

    // Shade point
    shade_point(p_payload, l_mesh_vertex, l_material);
}

[shader("closesthit")]
void tile_shadow_closest_hit_shader(inout pl_shadow_ray_payload_t p_payload, in BuiltInTriangleIntersectionAttributes p_attribs)
{
    p_payload.m_hit = true;
}
