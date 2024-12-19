// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Optimization ideas:
//  - Disable three POVs blending and/or parallax on very distant impostors
//  - Tile to UV mapping (dividing by division count) should be done as early as possible

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting.hlsl"

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

struct ps_input_t
{
    float4 m_position           : SV_POSITION;
    float3 m_impostor_center_ws : POSITION0;
    float3 m_vertex_camera_ws   : POSITION1;

    nointerpolation uint2  m_tile_0  : TEXCOORD0;
    nointerpolation uint2  m_tile_1  : TEXCOORD1;
    nointerpolation uint2  m_tile_2  : TEXCOORD2;
    nointerpolation float3 m_weights : TEXCOORD3;

    float4 m_uv_and_frame_0 : TEXCOORD4;
    float4 m_uv_and_frame_1 : TEXCOORD5;
    float4 m_uv_and_frame_2 : TEXCOORD6;

    uint m_instance_id : SV_InstanceID;
};

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_impostor_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// VS
//-----------------------------------------------------------------------------
float3 uv_to_hemisphere_octahedron(float2 p_uv)
{
    float3 l_position = float3(p_uv.x - p_uv.y, 0.f, -1.f + p_uv.x + p_uv.y);
    float3 l_abs_position = abs(l_position);
    l_position.y = 1.f - l_abs_position.x - l_abs_position.z;
    return normalize(l_position);
}

//-----------------------------------------------------------------------------
float2 hemisphere_octahedron_to_uv(float3 p_view_vector)
{
    // Restrict to top hemisphere
    float3 l_hemisphere_vector = p_view_vector;
    l_hemisphere_vector.y = max(l_hemisphere_vector.y, 0.f);

    float3 l_octant = float3(sign(l_hemisphere_vector.x), sign(l_hemisphere_vector.y), sign(l_hemisphere_vector.z));

    float l_sum = dot(p_view_vector, l_octant);
    float3 l_octahedron = p_view_vector / l_sum;

    float2 l_uv = float2(l_octahedron.x + l_octahedron.z, l_octahedron.z - l_octahedron.x);
    return saturate(l_uv * 0.5f + float2(0.5f, 0.5f));
}

//-----------------------------------------------------------------------------
void get_basis_xy(float3 p_up, float3 p_forward, out float3 p_x, out float3 p_y)
{
    //p_up = abs(p_forward.z) < 0.999f ? float3(0,0,1) : float3(1,0,0);
    p_x = normalize(cross(p_up,      p_forward));
    p_y = normalize(cross(p_forward, p_x      ));
}

//-----------------------------------------------------------------------------
float3x3 get_transform_matrix(sb_impostor_instance_t p_instance)
{
    float l_sin = 0;
    float l_cos = 1;
    sincos(p_instance.m_angle, l_sin, l_cos);

    float3 l_normal   = p_instance.m_up_vector;
    float3 l_binormal = normalize(cross(float3(1,0,0), l_normal));
    float3 l_tangent  = normalize(cross(l_normal,      l_binormal));

    float3x3 l_combined_transform;
    l_combined_transform._11_12_13 = (l_cos * l_tangent - l_sin * l_binormal) * p_instance.m_scale;
    l_combined_transform._21_22_23 = l_normal * p_instance.m_scale;
    l_combined_transform._31_32_33 = (l_sin * l_tangent + l_cos * l_binormal) * p_instance.m_scale;
    return l_combined_transform;
}

//-----------------------------------------------------------------------------
float2 get_projected_uv(float3 p_octahedral_forward, float3 p_octahedral_x, float3 p_octahedral_y, float3 p_impostor_center_ws, float3 p_vertex_camera_ws, float p_model_radius)
{
    // Compute intersection of octahedron facing impostor plane with ray passing through camera impostor vertex
    //  Reference: https://paulbourke.net/geometry/pointlineplane/
    float3 l_plane_normal = p_octahedral_forward;
    float3 l_plane_point  = p_impostor_center_ws; // Pivot is the same on both planes
    float3 l_line_p1      = p_vertex_camera_ws;
    float3 l_line_p0      = float3(0,0,0);

    float l_numerator   = dot(l_plane_normal, l_plane_point - l_line_p0);
    float l_denominator = dot(l_plane_normal, l_line_p1     - l_line_p0);
    float l_unit        = abs(l_denominator) > 0.0001 ? (l_numerator / l_denominator) : 0.0;

    float3 l_intersection = (l_line_p0 + l_unit * (l_line_p1 - l_line_p0)) - p_impostor_center_ws;

    float2 l_proj_uv = float2( dot(p_octahedral_x, l_intersection),
                              -dot(p_octahedral_y, l_intersection));

    return (l_proj_uv/(2 * p_model_radius)) + 0.5;
}

//-----------------------------------------------------------------------------
float4 get_tile_uv_and_frame(uint2 p_tile, sb_impostor_instance_t p_instance, sb_impostor_item_t p_item, float3 p_impostor_center_ws, float3 p_vertex_camera_ws)
{
    // Compute octahedral basis and projected UV
    float3 l_octahedral_forward = -uv_to_hemisphere_octahedron(p_tile / (p_item.m_octahedron_divisions - 1));

    float3x3 l_transform = get_transform_matrix(p_instance);
    l_octahedral_forward = mul(l_octahedral_forward, l_transform);
    l_octahedral_forward = normalize(l_octahedral_forward);

    float3 l_octahedral_x, l_octahedral_y;
    get_basis_xy(p_instance.m_up_vector, l_octahedral_forward, l_octahedral_x, l_octahedral_y);

    float2 l_uv = get_projected_uv(l_octahedral_forward, l_octahedral_x, l_octahedral_y, p_impostor_center_ws, p_vertex_camera_ws, p_item.m_bounding_sphere.w * p_instance.m_scale);

    // Use depth to compute UV parallax and fetch albedo/alpha
    float3 l_vertex_dir = -normalize(p_vertex_camera_ws);
    float2 l_frame = float2( dot(l_octahedral_x, l_vertex_dir),
                            -dot(l_octahedral_y, l_vertex_dir));

    return float4(l_uv, l_frame);
}

//-----------------------------------------------------------------------------
ps_input_t vs_main(uint p_vertex_id   : SV_VertexID,
                   uint p_instance_id : SV_InstanceID)
{
    StructuredBuffer<uint> l_sorted_instance_list = ResourceDescriptorHeap[g_push_constants.m_sorted_instance_buffer_srv];
    uint l_instance_id = l_sorted_instance_list[p_instance_id + g_push_constants.m_start_instance_location];

    StructuredBuffer<sb_impostor_instance_t> l_instance_list = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_srv];
    StructuredBuffer<sb_impostor_item_t> l_item_list = ResourceDescriptorHeap[g_push_constants.m_item_buffer_srv];
    sb_impostor_instance_t l_instance = l_instance_list[l_instance_id];
    sb_impostor_item_t l_item = l_item_list[l_instance.m_item_idx];

    // Impostor pivot/center position in local world space (relative to the camera)
    float3x3 l_transform = get_transform_matrix(l_instance);

    float3 l_center_ws = l_instance.m_position + mul(l_item.m_bounding_sphere.xyz, l_transform);
    float3 l_camera_forward = -normalize(l_center_ws);

    // Compute impostor vertex position when using camera direction
    float3 l_camera_x, l_camera_y;
    get_basis_xy(l_instance.m_up_vector, l_camera_forward, l_camera_x, l_camera_y);

    float2 l_vertex_offset = get_fullscreen_quad_position(p_vertex_id).xy * l_item.m_bounding_sphere.w * l_instance.m_scale;
    float3 l_vertex_camera_ws = l_center_ws +
                                l_vertex_offset.x * l_camera_x +
                                l_vertex_offset.y * l_camera_y;

    // Compute tiles/weights
    l_camera_forward = mul(l_transform, l_camera_forward);

    float2 l_hemisphere_uv = hemisphere_octahedron_to_uv(l_camera_forward);
    float2 l_tile_pos = l_hemisphere_uv * (float)(l_item.m_octahedron_divisions - 1);
    float2 l_tile_fract = frac(l_tile_pos);
    uint2 l_tile_0 = (uint2)l_tile_pos;
    uint2 l_tile_1 = l_tile_0 + ((l_tile_fract.x > l_tile_fract.y) ? uint2(1, 0) : uint2(0, 1));
    uint2 l_tile_2 = l_tile_0 + uint2(1, 1);

    l_tile_0 = clamp(l_tile_0, 0, l_item.m_octahedron_divisions - 1);
    l_tile_1 = clamp(l_tile_1, 0, l_item.m_octahedron_divisions - 1);
    l_tile_2 = clamp(l_tile_2, 0, l_item.m_octahedron_divisions - 1);

    float3 l_weights = float3(min(1.f - l_tile_fract.x, 1.f - l_tile_fract.y),
                              abs(l_tile_fract.x - l_tile_fract.y),
                              min(l_tile_fract.x, l_tile_fract.y));

    // View space
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    float4 l_pos_vs = mul(float4(l_vertex_camera_ws,1), l_camera.m_view_local);

    // Clip space
    float4 l_pos_cs = mul(l_pos_vs, l_camera.m_proj);

    ps_input_t l_result;
    l_result.m_position           = l_pos_cs;
    l_result.m_impostor_center_ws = l_center_ws;
    l_result.m_vertex_camera_ws   = l_vertex_camera_ws;

    l_result.m_tile_0  = l_tile_0;
    l_result.m_tile_1  = l_tile_1;
    l_result.m_tile_2  = l_tile_2;
    l_result.m_weights = l_weights;

    l_result.m_uv_and_frame_0 = get_tile_uv_and_frame(l_tile_0, l_instance, l_item, l_center_ws, l_vertex_camera_ws);
    l_result.m_uv_and_frame_1 = get_tile_uv_and_frame(l_tile_1, l_instance, l_item, l_center_ws, l_vertex_camera_ws);
    l_result.m_uv_and_frame_2 = get_tile_uv_and_frame(l_tile_2, l_instance, l_item, l_center_ws, l_vertex_camera_ws);

    l_result.m_instance_id = l_instance_id;
    return l_result;
}

//-----------------------------------------------------------------------------
// PS
//-----------------------------------------------------------------------------
void get_projected_data(out float4 p_albedo_alpha, out float4 p_normal_depth, uint2 p_tile, float4 p_uv_and_frame, sb_impostor_instance_t p_instance, sb_impostor_item_t p_item, float3 p_impostor_center_ws, float3 p_vertex_camera_ws)
{
    //  HACK: We should be perspective correcting the UV and frame, but it's not noticeable. Uncomment the following
    // to use UV and frame information projected directly in here, as it is the ground truth, but also slower
    //p_uv_and_frame = get_tile_uv_and_frame(p_tile, p_instance, p_item, p_impostor_center_ws, p_vertex_camera_ws);

    SamplerState l_sampler = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];
    float2 l_uv_tile_normal_depth = (p_tile + saturate(p_uv_and_frame.xy)) / p_item.m_octahedron_divisions;
    float depth = bindless_tex2d_sample(p_item.m_normal_depth_srv, l_sampler, l_uv_tile_normal_depth).w;

    // TODO: This might prove useful to be moved to a bake time parameter
    static const float g_depth_scale = 1.f;
    float2 l_parallax_uv = p_uv_and_frame.zw * (depth - 0.5) * g_depth_scale + p_uv_and_frame.xy;
    float2 l_uv_tile = (p_tile + saturate(l_parallax_uv)) / p_item.m_octahedron_divisions;
    p_normal_depth = bindless_tex2d_sample(p_item.m_normal_depth_srv, l_sampler, l_uv_tile);
    p_albedo_alpha = bindless_tex2d_sample(p_item.m_albedo_alpha_srv, l_sampler, l_uv_tile);
}

//-----------------------------------------------------------------------------
float4 ps_main(ps_input_t p_input) : SV_TARGET
{
    StructuredBuffer<sb_impostor_instance_t> l_instance_list = ResourceDescriptorHeap[g_push_constants.m_instance_buffer_srv];
    StructuredBuffer<sb_impostor_item_t> l_item_list = ResourceDescriptorHeap[g_push_constants.m_item_buffer_srv];
    sb_impostor_instance_t l_instance = l_instance_list[p_input.m_instance_id];
    sb_impostor_item_t l_item = l_item_list[l_instance.m_item_idx];

    float4 l_albedo_alpha_0, l_albedo_alpha_1, l_albedo_alpha_2;
    float4 l_normal_depth_0, l_normal_depth_1, l_normal_depth_2;
    get_projected_data(l_albedo_alpha_0, l_normal_depth_0, p_input.m_tile_0, p_input.m_uv_and_frame_0, l_instance, l_item, p_input.m_impostor_center_ws, p_input.m_vertex_camera_ws);
    get_projected_data(l_albedo_alpha_1, l_normal_depth_1, p_input.m_tile_1, p_input.m_uv_and_frame_1, l_instance, l_item, p_input.m_impostor_center_ws, p_input.m_vertex_camera_ws);
    get_projected_data(l_albedo_alpha_2, l_normal_depth_2, p_input.m_tile_2, p_input.m_uv_and_frame_2, l_instance, l_item, p_input.m_impostor_center_ws, p_input.m_vertex_camera_ws);

    float4 l_albedo_alpha = l_albedo_alpha_0 * p_input.m_weights.x +
                            l_albedo_alpha_1 * p_input.m_weights.y +
                            l_albedo_alpha_2 * p_input.m_weights.z;

    float4 l_normal_depth = l_normal_depth_0 * p_input.m_weights.x +
                            l_normal_depth_1 * p_input.m_weights.y +
                            l_normal_depth_2 * p_input.m_weights.z;

#if defined(MB_ALPHA_TEST)
    float l_alpha = l_albedo_alpha.a;
    // Extreme close ups cause BC7 compression to affect alpha (as it is correlated with RGB data), smoothstepping it helps a bit by expanding range
    //l_alpha = smoothstep(0, 1.0, l_alpha);
    if(alpha_test(l_alpha, p_input.m_vertex_camera_ws))
    {
        discard;
    }
#else
    clip(l_albedo_alpha.a - 0.7);
#endif

    float3x3 l_transform = get_transform_matrix(l_instance);
    float3 l_normal_ws = mul(l_normal_depth.xyz * 2.f - 1.f, l_transform);
    l_normal_ws = normalize(l_normal_ws);

    // TODO: Currently hardcoded, might add to baked data if really necessary
    float l_metallic = 0;
    float l_base_color_factor = 1.f;
    float l_roughness = 0.f;
    float l_ao = 1.f;

    float3 l_base_color = gamma_to_linear(l_albedo_alpha.rgb * l_base_color_factor);
    float3 l_diffuse_reflectance = base_color_to_diffuse_reflectance(l_base_color, l_metallic);
    float3 l_specular_f0 = base_color_to_specular_f0(l_base_color, l_metallic);

    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    float3 l_planet_normal = normalize(p_input.m_vertex_camera_ws + l_camera.m_camera_pos);

    float3 l_direct_lighting = (float3)0;
    float3 l_indirect_lighting = (float3)0;
    calc_lighting(
        p_input.m_vertex_camera_ws,
        l_normal_ws,
        l_roughness,
        l_diffuse_reflectance,
        l_specular_f0,
        l_planet_normal,
        l_ao,
        ResourceDescriptorHeap[g_push_constants.m_light_list_cbv],
        0,                       // shadow_caster_count
        RAL_NULL_BINDLESS_INDEX, // shadow_caster_srv,
        g_push_constants.m_exposure_value,
        g_push_constants.m_dfg_texture_srv,
        g_push_constants.m_diffuse_ld_texture_srv,
        g_push_constants.m_specular_ld_texture_srv,
        g_push_constants.m_dfg_texture_size,
        g_push_constants.m_specular_ld_mip_count,
        g_push_constants.m_gsm_srv, 
        g_push_constants.m_gsm_camera_view_local_proj,
        l_direct_lighting,
        l_indirect_lighting);

    float3 l_final_color = l_direct_lighting + l_indirect_lighting;
    return float4(pack_lighting(l_final_color), 1);
}
