// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"
#include "../helper_shaders/mb_util_noise.hlsl"
#include "mb_instancing_common.hlsl"
#include "mb_lighting.hlsl"
#include "mb_grass_common.hlsl"

#define MB_ENABLE_GRASS_LOD
//#define MB_DEBUG_GRASS_LOD

struct grass_payload_t
{
    uint m_patch_count;
};

struct vertex_out_t
{
    float4 m_position_cs                    : SV_POSITION;
    float3 m_position_camera_local          : POSITION0;
    float3 m_normal_ws                      : NORMAL0;
    float3 m_ground_normal                  : NORMAL1;
    float3 m_root_position_camera_local     : POSITION1;
    float m_height                          : TEXCOORD0;
    float m_gradient                        : TEXCOORD1;
    nointerpolation float3 m_color          : COLOR0;
    nointerpolation float3 m_ground_color   : COLOR1;
#if defined(MB_DEBUG_GRASS_LOD)
    float3 m_debug_color                    : COLOR2;
#endif
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float3 m_proj_pos_curr                  : POSITION2;
    float3 m_proj_pos_prev                  : POSITION3;
#endif
};

struct ps_output_t
{
    float4 m_direct_lighting    : SV_TARGET0;
    float4 m_indirect_lighting  : SV_TARGET1;
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    float2 m_velocity           : SV_TARGET2;
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    uint m_entity_id            : SV_TARGET3;
#endif
};

ConstantBuffer<cb_push_grass_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

// The groupshared payload data to be exported from the amplification shader threadgroup to dispatched mesh shader threadgroups
groupshared grass_payload_t g_payload;

uint hash(uint x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

// Generates a [0, 1] float using a hash-based LCG
float rand(uint seed, uint index)
{
    // Mix seed and index
    seed ^= hash(index);
    seed ^= 0x9E3779B9u; // Constant used to disrupt linear sequences

    // Linear Congruential Generator (LCG)
    seed = seed * 1664525u + 1013904223u;

    // Convert to a float in the range [0, 1]
    return float(seed) / float(0xFFFFFFFFu);
}

// Generates a [0, 1] float using a hash-based LCG
float rand(uint seed, uint index1, uint index2)
{
    // Mix seed and indices
    seed ^= hash(index1);
    seed ^= hash(index2);
    seed ^= 0x9E3779B9u; // Constant used to disrupt linear sequences

    // Linear Congruential Generator (LCG)
    seed = seed * 1664525u + 1013904223u;

    // Convert to a float in the range [0, 1]
    return float(seed) / float(0xFFFFFFFFu);
}

// https://github.com/klejah/ResponsiveGrassDemo/blob/master/ResponsiveGrassDemo/shader/Grass/GrassUpdateForcesShader.cs#L67
void make_persistent_length(float3 v0, inout float3 v1, inout float3 v2, float height)
{
    // Persistent length
    float3 v01 = v1 - v0;
    float3 v12 = v2 - v1;
    float lv01 = length(v01);
    float lv12 = length(v12);

    float l1 = lv01 + lv12;
    float l0 = length(v2 - v0);
    float l = (2.0f * l0 + l1) / 3.0f; // http://steve.hollasch.net/cgindex/curves/cbezarclen.html

    float ldiff = height / l;
    v01 = v01 * ldiff;
    v12 = v12 * ldiff;
    v1 = v0 + v01;
    v2 = v1 + v12;
}

// Quadratic Bezier Curve
// La = (1 - t) * P0 + t * P1
// Lb = (1 - t) * P1 + t * P2
// B(t) = (1 - t) * La + t * Lb
float3 bezier(float3 p0, float3 p1, float3 p2, float t)
{
    float3 a = lerp(p0, p1, t);
    float3 b = lerp(p1, p2, t);
    return lerp(a, b, t);
}

// Derivative of Quadratic Bezier Curve
// B(t) = (1 - t)^2 * P0 + 2 * t * (1 - t) * P1 + t^2 * P2
// B'(t) = -2 * (1 - t) * P0 + 2 * P1 - 4 * t * P1 + 2 * t * P2
// B'(t) = 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)
float3 bezier_derivative(float3 p0, float3 p1, float3 p2, float t)
{
    return 2.0f * (1.0f - t) * (p1 - p0) + 2.0f * t * (p2 - p1);
}

float3 apply_wind(float time, float3 position, uint seed, uint blade_id,  float3 tangent, float3 bitangent, float wind_strength)
{
    float sin_wave = sin(position.x * 0.40 - time * 0.0020 + rand(seed, 31 - seed) * 3.0 + rand(seed, 31 - seed, blade_id) * 1.0);
    float cos_wave = cos(position.y * 0.35 - time * 0.0024 + rand(seed, 32 - seed) * 3.0 + rand(seed, 32 - seed, blade_id) * 1.0);

    return (sin_wave * tangent + cos_wave * bitangent) * g_push_constants.m_wind_scale * wind_strength;
}

// Calculate IBL without fake earth shadow to get rid of exploding ISA on AMD Radeon RX 6900 XT
float3 light_ibl_grass(float3 normal_ws,
                       float3 view_dir,
                       float roughness,
                       float ao,
                       float3 diffuse_reflectance,
                       float3 specular_f0,
                       float exposure_value,
                       uint dfg_texture_srv,
                       uint diffuse_ld_texture_srv,
                       uint specular_ld_texture_srv,
                       uint dfg_texture_size,
                       uint specular_ld_mip_count,
                       float3 planet_normal,
                       cb_light_list_t light_list,
                       float3x3 align_ground_rotation)
{
    float ibl_intensity_multiplier = ev100_to_luminance(exposure_value);

    // IBL is baked with identity rotation. Depending where camera position is - probe needs to be rotated.
    // Instead of rotating the probe we rotate the vectors
    normal_ws = mul(normal_ws, align_ground_rotation);
    view_dir = mul(view_dir, align_ground_rotation);

    // IBL diffuse
    float3 ibl_diffuse = evaluate_ibl_diffuse(normal_ws, view_dir, roughness, dfg_texture_srv, diffuse_ld_texture_srv) * diffuse_reflectance;
    ibl_diffuse *= ibl_intensity_multiplier;

    // IBL specular
    float3 ibl_specular = evaluate_ibl_specular(normal_ws, view_dir, roughness, dfg_texture_srv, dfg_texture_size, specular_ld_texture_srv, specular_ld_mip_count, specular_f0, 1);
    ibl_specular *= ibl_intensity_multiplier;

    // Compute specular AO
    float nov = saturate(dot(normal_ws, view_dir));
    float alpha = clamp_and_remap_roughness(roughness);
    float spec_ao = compute_specular_occlusion(nov, ao, alpha);

    // Fake Earth shadow BRDF
    float fake_earth_ibl_shadow = fake_earth_ibl_shadow_brdf(planet_normal, light_list);

    // Apply IBL
    float3 ibl = (ibl_diffuse * ao + ibl_specular * spec_ao) * fake_earth_ibl_shadow;

    return ibl;
}

void calc_lighting_grass(float3 position_ws_local,
                         float3 normal_ws,
                         float roughness,
                         float3 diffuse_reflectance,
                         float3 specular_f0,
                         float3 planet_normal,
                         float ao,
                         ConstantBuffer<cb_light_list_t> light_list,
                         uint shadow_caster_count,
                         uint shadow_caster_srv,
                         float exposure_value,
                         uint dfg_texture_srv,
                         uint diffuse_ld_texture_srv,
                         uint specular_ld_texture_srv,
                         uint dfg_texture_size,
                         uint specular_ld_mip_count,
                         uint global_shadow_map_srv,
                         float4x4 gsm_camera_view_local_proj,
                         float3x3 p_align_ground_rotation,
                         out float3 direct_lighting,
                         out float3 indirect_lighting)
{
    // Camera-local rendering
    float3 view_dir = normalize(-position_ws_local);

    // Direct lighting
    direct_lighting = light_direct_translucent(light_list, normal_ws, view_dir, roughness, diffuse_reflectance, specular_f0,
                                               position_ws_local,
                                               planet_normal,
                                               shadow_caster_count, shadow_caster_srv,
                                               global_shadow_map_srv, gsm_camera_view_local_proj);

    indirect_lighting = (float3)0;
#if ENABLE_IBL
    indirect_lighting = light_ibl_grass(normal_ws, view_dir, roughness, ao, diffuse_reflectance, specular_f0,
                                        exposure_value, dfg_texture_srv, diffuse_ld_texture_srv, specular_ld_texture_srv,
                                        dfg_texture_size, specular_ld_mip_count, planet_normal, light_list, p_align_ground_rotation);
#endif
}

[NumThreads(1, 1, 1)]
void as_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    StructuredBuffer<uint> count_buffer = ResourceDescriptorHeap[g_push_constants.m_patch_lod_count_buffer_srv];
    uint patch_count = count_buffer[g_push_constants.m_lod_level];
    g_payload.m_patch_count = patch_count;

    uint thread_group_count_x = patch_count;
    uint thread_group_count_y = 1;
    uint thread_group_count_z = 1;

    if (patch_count == 0)
    {
        // Can't use return here. Set thread group count to 0 to avoid "Non-Dominating DispatchMesh call" error.
        thread_group_count_x = 0;
        thread_group_count_y = 0;
        thread_group_count_z = 0;
    }

    DispatchMesh(thread_group_count_x, thread_group_count_y, thread_group_count_z, g_payload);
}

#define MB_MS_GROUP_SIZE 32
static const uint VERTEX_COUNT = 256;
static const uint PRIMITIVE_COUNT = 192;

// Mesh shaders can only output up to 256 vertices and use a thread group size of up to 128.
// *See: https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#amplification-shader-and-mesh-shader
// Therefore, we use N loops per thread here to generate certain amount of vertices/primitives in each thread group.
static const uint VERTEX_LOOP = (VERTEX_COUNT + MB_MS_GROUP_SIZE - 1) / MB_MS_GROUP_SIZE;
static const uint PRIMITIVE_LOOP = (PRIMITIVE_COUNT + MB_MS_GROUP_SIZE - 1) / MB_MS_GROUP_SIZE;

static const float GRASS_LEANING = 0.3f;

// Make the grass blade narrower at the top
static const float OFFSET_WEIGHT_0 = 1.0f;
static const float OFFSET_WEIGHT_1 = 0.7f;
static const float OFFSET_WEIGHT_2 = 0.3f;

static const float MAX_WIND_DISTANCE = 30.0f;

[NumThreads(MB_MS_GROUP_SIZE, 1, 1)]
[OutputTopology("triangle")]
void ms_main(uint group_thread_id : SV_GroupThreadID,
             uint3 group_id : SV_GroupID,
             in payload grass_payload_t patch_args,
             out indices uint3 triangles[PRIMITIVE_COUNT],
             out vertices vertex_out_t vertices[VERTEX_COUNT])
{
    // Get patch arguments
    StructuredBuffer<sb_grass_patch_argument_t> patch_arguments = ResourceDescriptorHeap[g_push_constants.m_patch_arguments_srv];
    sb_grass_patch_argument_t patch_argument = patch_arguments[group_id.x];
    float3 patch_position_camera_local = patch_argument.m_position_camera_local;
    float3 ground_normal = patch_argument.m_ground_normal;
    float3 ground_color = patch_argument.m_ground_color;
    float3 color = patch_argument.m_color;
    uint seed = patch_argument.m_random_seed;
    float blade_width = patch_argument.m_blade_width;
    float blade_height = patch_argument.m_blade_height;
    float patch_radius = patch_argument.m_patch_radius;
    uint blade_count = patch_argument.m_blade_count;

    // Set output
    uint vertex_count = blade_count * VERTICES_PER_BLADE;
    uint triangle_count = blade_count * TRIANGLES_PER_BLADE;
    SetMeshOutputCounts(vertex_count, triangle_count);

    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    float distance_to_camera = length(patch_position_camera_local);
#if defined(MB_ENABLE_GRASS_LOD)
    float blade_count_f = lerp(float(MAX_BLADE_COUNT), 2.0f, pow(saturate(distance_to_camera / (GRASS_LOD_END_DISTANCE * 1.05f)), 0.75f));
#else
    float blade_count_f = float(MAX_BLADE_COUNT);
#endif

    float wind_strength = saturate(1.0f - (distance_to_camera / MAX_WIND_DISTANCE));

    // Position the grass in a circle around the patch center, angled with the ground normal
    float3 tangent = normalize(cross(ground_normal, float3(0, 0, 1)));
    float3 bitangent = normalize(cross(tangent, ground_normal));

    for (uint i = 0; i < VERTEX_LOOP; ++i)
    {
        uint vertex_id = group_thread_id + MB_MS_GROUP_SIZE * i;

        if (vertex_id >= vertex_count)
        {
            break;
        }

        uint blade_id = vertex_id / VERTICES_PER_BLADE;
        uint vertex_id_local = vertex_id % VERTICES_PER_BLADE;

        float curr_blade_height = blade_height + rand(seed, blade_id, 20) * RANDOM_EXTRA_HEIGHT_SCALE;

        // Get blade direction
        float blade_direction_angle = 2.0f * M_PI * rand(seed, 4, blade_id);
        float3 blade_direction = cos(blade_direction_angle) * tangent + sin(blade_direction_angle) * bitangent;

        float3 blade_offset = (rand(seed, 23 - seed, blade_id) - 0.5) * tangent + (rand(seed, 24 - seed, blade_id) - 0.5) * bitangent;
        blade_offset *= patch_radius;

        // Direction to make grass point away from the camera
        float3 patch_view_direction = normalize(patch_position_camera_local + blade_offset);
        float3 point_away_camera_tangent =   dot(patch_view_direction, tangent)   * tangent;
        float3 point_away_camera_bitangent = dot(patch_view_direction, bitangent) * bitangent;
        float3 point_away_camera = normalize(point_away_camera_tangent + point_away_camera_bitangent);

        float3 v0 = blade_offset;
        float3 v1 = v0 + (ground_normal + (point_away_camera * saturate(3.0 - distance_to_camera))) * curr_blade_height * GRASS_LEANING;
        float3 v2 = v1 + (blade_offset * 10.0f + ground_normal + (point_away_camera * saturate(3.0 - distance_to_camera))) * curr_blade_height * GRASS_LEANING;

#if defined(MB_WIND)
        float3 position = (patch_position_camera_local + blade_offset) + camera.m_camera_pos_frac_100;

#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
        // Duplicate control points and offset by previous frame time, to get object motion vectors
        float3 v0_prev = v0;
        float3 v1_prev = v1;
        float3 v2_prev = v2;

        v2_prev += apply_wind(g_push_constants.m_time_prev, position, seed, blade_id, tangent, bitangent, wind_strength);

        make_persistent_length(v0_prev, v1_prev, v2_prev, curr_blade_height);
#endif // MB_RENDER_VELOCITY_PASS_ENABLED

        v2 += apply_wind(g_push_constants.m_time, position, seed, blade_id, tangent, bitangent, wind_strength);
#endif // MB_WIND

        // To animate the grass, we move v2 which modifies the length of the Bezier curve.
        // To preserve the length of the curve, we use this function to modify v1 and v2 to retain the length of the curve.
        make_persistent_length(v0, v1, v2, curr_blade_height);

        float curr_blade_width = blade_width;

        // Adjust blade width based on blade count to maintain consistent ground coverage
        curr_blade_width *= sqrt(MAX_BLADE_COUNT / blade_count_f);

        vertex_out_t vertex;
        vertex.m_height = blade_height;
        vertex.m_root_position_camera_local = v0 + patch_position_camera_local;

        // Side direction is perpendicular to the blade direction, potentially pointing towards one side of the grass blade
        float3 side_direction = normalize(float3(blade_direction.y, -blade_direction.x, 0));

        float3 vertex_offset = ((vertex_id_local & 1) ? 1.0f : -1.0f) * curr_blade_width * side_direction;

        v0 += vertex_offset * OFFSET_WEIGHT_0;
        v1 += vertex_offset * OFFSET_WEIGHT_1;
        v2 += vertex_offset * OFFSET_WEIGHT_2;

        float t = (vertex_id_local >> 1) / float(VERTICES_PER_BLADE_EDGE - 1);
        vertex.m_gradient = t;
        vertex.m_position_camera_local = bezier(v0, v1, v2, t) + patch_position_camera_local;
        vertex.m_normal_ws = cross(normalize(bezier_derivative(v0, v1, v2, t)), side_direction);
        vertex.m_position_cs = mul(float4(vertex.m_position_camera_local, 1), camera.m_view_proj_local);
        vertex.m_ground_normal = ground_normal;
        vertex.m_color = color;
        vertex.m_ground_color = ground_color;
#if defined(MB_DEBUG_GRASS_LOD)
        switch (blade_count % 4)
        {
            case 0: vertex.m_debug_color = float3(1, 0, 0); break;
            case 1: vertex.m_debug_color = float3(1, 1, 0); break;
            case 2: vertex.m_debug_color = float3(0, 0, 1); break;
            case 3: vertex.m_debug_color = float3(0, 1, 0); break;
        }
#endif
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
#if defined(MB_WIND)
        v0_prev += vertex_offset * OFFSET_WEIGHT_0;
        v1_prev += vertex_offset * OFFSET_WEIGHT_1;
        v2_prev += vertex_offset * OFFSET_WEIGHT_2;
        float3 vertex_prev = bezier(v0_prev, v1_prev, v2_prev, t) + patch_position_camera_local;
#else
        float3 vertex_prev = vertex.m_position_camera_local;
#endif

        float4 pos_cs_prev = mul(float4(vertex_prev, 1), camera.m_view_proj_local_prev);
        vertex.m_proj_pos_curr = vertex.m_position_cs.xyw;
        vertex.m_proj_pos_prev = pos_cs_prev.xyw;
#endif

        vertices[vertex_id] = vertex;
    }

    for (uint j = 0; j < PRIMITIVE_LOOP; ++j)
    {
        uint triangle_id = group_thread_id + MB_MS_GROUP_SIZE * j;

        if (triangle_id >= triangle_count)
        {
            break;
        }

        uint blade_id = triangle_id / TRIANGLES_PER_BLADE;
        uint triangle_id_local = triangle_id % TRIANGLES_PER_BLADE;

        uint offset = blade_id * VERTICES_PER_BLADE + 2 * (triangle_id_local / 2);

        // v2 ---- v3
        //    |\ |
        //    | \|
        // v0 ---- v1
        uint3 triangle_indices = (triangle_id_local & 1) == 0 ? uint3(0, 2, 1) : uint3(3, 1, 2);

        triangles[triangle_id] = offset + triangle_indices;
    }
}

ps_output_t ps_main(vertex_out_t input, bool front_face : SV_IsFrontFace)
{
    float distance_to_camera = length(input.m_position_camera_local);
    float blend_with_ground_factor = 1.0f - saturate((distance_to_camera - g_push_constants.m_ground_blend_start_distance) * g_push_constants.m_ground_blend_range_reciprocal);

    float3 base_color = gamma_to_linear(lerp(input.m_ground_color, input.m_color, input.m_gradient * input.m_gradient * blend_with_ground_factor));

    const float roughness = 0.65f;
    const float metallic = 0.0f;
    const float ao = 1.0f;

    float3 diffuse_reflectance = base_color_to_diffuse_reflectance(base_color, metallic);
    float3 specular_f0 = base_color_to_specular_f0(base_color, metallic);

    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    float3 planet_normal = normalize(input.m_position_camera_local + camera.m_camera_pos);

    float3 normal_ws = normalize(input.m_normal_ws);
    normal_ws = front_face ? normal_ws : -normal_ws;
    normal_ws = lerp(planet_normal, normal_ws + planet_normal * 3.0, blend_with_ground_factor);
    normal_ws = normalize(normal_ws);

    float3 direct_lighting = (float3)0;
    float3 indirect_lighting = (float3)0;

    // Self shadow
    float frag_height = dot(input.m_position_camera_local - input.m_root_position_camera_local, input.m_ground_normal);
    float self_shadow = saturate(pow(frag_height / input.m_height, 2.0f));

    calc_lighting_grass(input.m_position_camera_local,
                        normal_ws,
                        roughness,
                        diffuse_reflectance,
                        specular_f0 * self_shadow, // reduce specular intensity in self shadows
                        planet_normal,
                        ao,
                        ResourceDescriptorHeap[g_push_constants.m_light_list_cbv],
                        g_push_constants.m_shadow_caster_count,
                        g_push_constants.m_shadow_caster_srv,
                        g_push_constants.m_exposure_value,
                        g_push_constants.m_dfg_texture_srv,
                        g_push_constants.m_diffuse_ld_texture_srv,
                        g_push_constants.m_specular_ld_texture_srv,
                        g_push_constants.m_dfg_texture_size,
                        g_push_constants.m_specular_ld_mip_count,
                        g_push_constants.m_gsm_srv,
                        g_push_constants.m_gsm_camera_view_local_proj,
                        (float3x3)camera.m_align_ground_rotation,
                        direct_lighting,
                        indirect_lighting);

    // Self shadow
    self_shadow = self_shadow * 0.5 + 0.5;
    self_shadow = lerp(1.0f, self_shadow, blend_with_ground_factor);
    direct_lighting *= self_shadow;

#if defined(MB_DEBUG_GRASS_LOD)
    direct_lighting *= input.m_debug_color;
    indirect_lighting = 0;
#endif

#if defined(MB_GRASS_DEBUG_CULLING_DISTANCE)
    direct_lighting.yz = float2(0, 0);
    indirect_lighting = 0;
#endif

    ps_output_t ps_output = (ps_output_t)0;
    ps_output.m_direct_lighting = float4(pack_lighting(direct_lighting), 1.0f);
    ps_output.m_indirect_lighting = float4(pack_lighting(indirect_lighting), 0.0f);

    // Pack lighting
#if defined(MB_RENDER_VELOCITY_PASS_ENABLED)
    ps_output.m_velocity = get_motion_vector_without_jitter(float2(camera.m_resolution_x, camera.m_resolution_y), input.m_proj_pos_curr, input.m_proj_pos_prev, camera.m_jitter, camera.m_jitter_prev);
#endif
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    ps_output.m_entity_id = 0;
#endif

    return ps_output;
}
