// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_lighting_common.hlsl"
#include "../shared_shaders/mb_shared_common.hlsl"

#define RENDER_DEBUG_SHAPE_SPHERE   0
#define RENDER_DEBUG_SHAPE_RECT     1
#define RENDER_DEBUG_SHAPE_CUBE     2
#define RENDER_DEBUG_SHAPE_PYRAMID  3

static float2 m_quad_vertices[6] =
{
    float2(0.0, 0.0),
    float2(0.0, 1.0),
    float2(1.0, 1.0),
    float2(0.0, 0.0),
    float2(1.0, 1.0),
    float2(1.0, 0.0)
};

static float2 m_quad_vertices_strip[4] =
{
    float2(0.0, 0.0),
    float2(0.0, 1.0),
    float2(1.0, 0.0),
    float2(1.0, 1.0)
};

static float3 m_pyramid_vertices[18] =
{
    float3(-0.5, 0.5, 1.0), float3(-0.5, -0.5, 1.0), float3( 0.5, -0.5, 1.0),
    float3(-0.5, 0.5, 1.0), float3( 0.5, -0.5, 1.0), float3( 0.5,  0.5, 1.0),
    float3( 0.0, 0.0, 0.0), float3(-0.5,  0.5, 1.0), float3(-0.5, -0.5, 1.0),
    float3( 0.0, 0.0, 0.0), float3( 0.5, -0.5, 1.0), float3( 0.5,  0.5, 1.0),
    float3( 0.0, 0.0, 0.0), float3( 0.5,  0.5, 1.0), float3(-0.5,  0.5, 1.0),
    float3( 0.0, 0.0, 0.0), float3( 0.5, -0.5, 1.0), float3(-0.5, -0.5, 1.0),
};

struct ps_input_t
{
    float4 m_position_cs      : SV_POSITION;
    uint m_entity_id          : TEXCOORD0;
    float3 m_color            : COLOR0;
};

struct ps_output_t
{
    float4 m_direct_lighting    : SV_TARGET0;
    float4 m_indirect_lighting  : SV_TARGET1;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    uint m_entity_id            : SV_TARGET3;
#endif
};

ConstantBuffer<cb_push_gltf_t>  g_push_constants : register(REGISTER_PUSH_CONSTANTS);

ps_input_t vs_gpu_instancing(   uint p_vertex_id    : SV_VertexID,
                                uint p_instance_id : SV_InstanceID)
{
    // Get render instance
    StructuredBuffer<sb_render_instance_t> l_render_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_render_instance_buffer_srv];
    sb_render_instance_t l_render_instance = l_render_instance_buffer[p_instance_id + g_push_constants.m_render_instance_buffer_offset];

    // Get render item info
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_render_instance.m_render_item_idx];

    // Get material
    StructuredBuffer<sb_debug_shape_material_t> l_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_debug_shape_material_t l_material = l_material_list[l_render_item.m_material_index];

    StructuredBuffer<sb_debug_shape_instance_data_t> l_instance_data_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_material.m_instance_data_srv)];
    sb_debug_shape_instance_data_t l_instance_data = l_instance_data_list[l_render_instance.m_user_data];

    float4 l_pos_ls = 0;
    if (l_material.m_type == RENDER_DEBUG_SHAPE_SPHERE)
    {
        const uint l_sphere_size_x = l_material.m_sphere_size_in_quads + l_material.m_sphere_size_in_quads;
        const uint l_sphere_size_y = l_material.m_sphere_size_in_quads;

        // Get quad coords
        const uint l_verts_in_quad = 2 * 3; // two triangles for each quad on the sphere
        uint l_quad_id = p_vertex_id / l_verts_in_quad;
        uint l_quad_x = l_quad_id % l_sphere_size_x;
        uint l_quad_y = l_quad_id / l_sphere_size_x;

        // Get vertex id in quad
        uint l_vert_id = p_vertex_id % l_verts_in_quad;

        // Convert quad to spherical coords
        float l_phi = 2.0f * M_PI * (float)l_quad_x / (float)l_sphere_size_x;
        float l_theta = M_PI * (float)l_quad_y / (float)l_sphere_size_y;

        // Check fo degenerate triangles at the sphere's poles
        bool l_degenerate_triangle = false;
        if (l_theta == 0 && l_vert_id > 2)
        {
            l_degenerate_triangle = true;
        }
        if (abs(l_theta + M_PI / (float)l_sphere_size_y - M_PI) < 0.001 && l_vert_id <= 2)
        {
            l_degenerate_triangle = true;
        }

        // Offset each vertex of a quad separately
        l_phi += m_quad_vertices[l_vert_id].x * 2.0f * M_PI / (float)l_sphere_size_x;
        l_theta += m_quad_vertices[l_vert_id].y * M_PI / (float)l_sphere_size_y;

        // Convert to Cartesian coords
        float r = 1.0f;
        l_pos_ls.x = r * cos(l_phi) * sin(l_theta);
        l_pos_ls.y = r * sin(l_phi) * sin(l_theta);
        l_pos_ls.z = r * cos(l_theta);
        l_pos_ls.w = 1.0f;

        // Clip degenerate triangles
        l_pos_ls = l_degenerate_triangle ? 0 : l_pos_ls;
    }
    else if (l_material.m_type == RENDER_DEBUG_SHAPE_RECT)
    {
        if (p_vertex_id < 4)
        {
            float2 l_vetrex = m_quad_vertices_strip[p_vertex_id] * 2.0f - 1.0f;
            l_pos_ls = float4(  l_vetrex.x,
                                0,
                                l_vetrex.y,
                                1.0);
        }
    }
    else if (l_material.m_type == RENDER_DEBUG_SHAPE_CUBE)
    {
        uint l_tmp = (uint)1 << p_vertex_id;
        l_pos_ls = float4(  (0x287a & l_tmp) != 0,
                            (0x02af & l_tmp) != 0,
                            (0x31e3 & l_tmp) != 0,
                            1.0);
        l_pos_ls.xyz = l_pos_ls.xyz * 2.0f - 1.0f;
    }
    else if (l_material.m_type == RENDER_DEBUG_SHAPE_PYRAMID)
    {
        l_pos_ls.xyz = m_pyramid_vertices[p_vertex_id];
        l_pos_ls.w = 1;
    }

    // Unpack input data
    ConstantBuffer<cb_camera_t> l_camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];

    // Positions
    float3 l_pos_ws = mul(float4(l_pos_ls.xyz, 1.0f), l_render_instance.m_transform);
    float4 l_pos_vs = mul(float4(l_pos_ws, 1.0f), l_camera.m_view_local);
    float4 l_pos_cs = mul(l_pos_vs, l_camera.m_proj);

    // Vertex shader output
    ps_input_t l_result;
    l_result.m_position_cs = l_pos_cs;
    l_result.m_color = l_instance_data.m_color.xyz;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_result.m_entity_id = l_render_instance.m_entity_id;
#endif

    return l_result;
}

void ps_shadow_pass(ps_input_t p_input)
{
}

void ps_visibility_pass()
{
}

void ps_impostor_data_pass()
{
}

ps_output_t ps_main(ps_input_t p_input)
{

    ps_output_t l_ps_output;
    l_ps_output.m_direct_lighting   = float4(p_input.m_color, 1.0);
    l_ps_output.m_indirect_lighting = 0;
#if defined(MB_RENDER_SELECTION_PASS_ENABLED)
    l_ps_output.m_entity_id         = pack_entity_id(p_input.m_entity_id);
#endif

    // Pack lighting
    l_ps_output.m_direct_lighting.rgb = pack_lighting(l_ps_output.m_direct_lighting.rgb);
    l_ps_output.m_indirect_lighting.rgb = pack_lighting(l_ps_output.m_indirect_lighting.rgb);

    return l_ps_output;
}
