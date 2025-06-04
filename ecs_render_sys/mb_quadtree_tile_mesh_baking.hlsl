// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "../helper_shaders/mb_quadtree_common.hlsl"

// CBV
ConstantBuffer<cb_push_tile_mesh_baking_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

float3 get_position(RWStructuredBuffer<float3> p_buffer, uint p_tile_index, uint2 p_vertex_index, float3 p_offset)
{
    uint l_data_index = p_tile_index * g_push_constants.m_num_vertices + p_vertex_index.y * g_push_constants.m_tile_vertex_resolution + p_vertex_index.x;
    float3 l_position = p_buffer[l_data_index] + p_offset;
    return l_position;
}

[numthreads(TILE_MESH_BAKING_THREAD_GROUP_SIZE, TILE_MESH_BAKING_THREAD_GROUP_SIZE, 1)]
void cs_main(uint p_group_index : SV_GroupIndex, uint3 p_group_id : SV_GroupID)
{
    if (p_group_index >= g_push_constants.m_num_vertices)
    {
        return;
    }

    // Outputs
    RWStructuredBuffer<float3> l_position_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_position_buffer_uav];
    RWStructuredBuffer<float3> l_normal_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_normal_buffer_uav];

    // Get per tile "instance" data
    StructuredBuffer<sb_tile_instance_to_bake_t> l_instance_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_instance_buffer_srv];
    sb_tile_instance_to_bake_t l_instance = l_instance_buffer[p_group_id.x];

    // Output data index
    uint l_data_index = l_instance.m_tile_index * g_push_constants.m_num_vertices + p_group_index;

    // Get heightmap array
    Texture2DArray l_heightmap_array = ResourceDescriptorHeap[g_push_constants.m_tile_heightmap_index];

    // Get tile position
    uint l_vertex_resolution_minus_one = g_push_constants.m_tile_vertex_resolution - 1;
    uint l_tile_position_x = p_group_index % g_push_constants.m_tile_vertex_resolution;
    uint l_tile_position_y = (p_group_index - l_tile_position_x) / g_push_constants.m_tile_vertex_resolution;
    float2 l_tile_position = float2(l_tile_position_x / (float)l_vertex_resolution_minus_one, l_tile_position_y / (float)l_vertex_resolution_minus_one);

    float3 l_position_ls = sample_terrain_position( l_instance,
                                                    l_tile_position,
                                                    g_push_constants.m_elevation_tile_resolution,
                                                    g_push_constants.m_elevation_tile_border,
                                                    l_heightmap_array);

    // Add horizontal displacement
#if defined(HORIZONTAL_DISPLACEMENT)
    RWStructuredBuffer<float3> l_undisplaced_position_buffer = ResourceDescriptorHeap[g_push_constants.m_tile_undisplaced_position_buffer_uav];
    l_undisplaced_position_buffer[l_data_index] = l_position_ls;

    if (l_instance.m_child_index_in_parent == 0xFFFFFFFF)
    {
        // No horizontal displacement
    }
    else
    {
        // Compute terrain normal by sampling 4 points around
        float3 l_position_0 = (float3)0;
        float3 l_position_1 = (float3)0;
        float3 l_position_2 = (float3)0;
        float3 l_position_3 = (float3)0;
        float3 l_position_4 = (float3)0;
        if (l_instance.m_child_index_in_parent == 0) // Top left
        {
            // Get vertex index in parent tile
            uint2 l_vertex_index = uint2(l_tile_position_x >> 1, (l_tile_position_y + g_push_constants.m_tile_vertex_resolution) >> 1);

            l_position_0 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index, float3(0, 0, 0));
            l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(1, 0), float3(0, 0, 0));

            if (l_vertex_index.y == l_vertex_resolution_minus_one)
            {
                l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_y_index, uint2(l_vertex_index.x, 1), l_instance.m_parent_neighbour_y_offset);
            }
            else
            {
                l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(0, 1), float3(0, 0, 0));
            }

            if (l_vertex_index.x == 0)
            {
                l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_x_index, uint2(l_vertex_resolution_minus_one - 1, l_vertex_index.y), l_instance.m_parent_neighbour_x_offset);
            }
            else
            {
                l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x - 1, l_vertex_index.y), float3(0, 0, 0));
            }

            l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x, l_vertex_index.y - 1), float3(0, 0, 0));
        }
        else if (l_instance.m_child_index_in_parent == 1) // Top right
        {
            // Get vertex index in parent tile
            uint2 l_vertex_index = uint2((l_tile_position_x + g_push_constants.m_tile_vertex_resolution) >> 1, (l_tile_position_y + g_push_constants.m_tile_vertex_resolution) >> 1);

            l_position_0 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index, float3(0, 0, 0));

            if (l_vertex_index.x == l_vertex_resolution_minus_one)
            {
                l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_x_index, uint2(1, l_vertex_index.y), l_instance.m_parent_neighbour_x_offset);
            }
            else
            {
                l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(1, 0), float3(0, 0, 0));
            }

            if (l_vertex_index.y == l_vertex_resolution_minus_one)
            {
                l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_y_index, uint2(l_vertex_index.x, 1), l_instance.m_parent_neighbour_y_offset);
            }
            else
            {
                l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(0, 1), float3(0, 0, 0));
            }

            l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x - 1, l_vertex_index.y), float3(0, 0, 0));
            l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x, l_vertex_index.y - 1), float3(0, 0, 0));
        }
        else if (l_instance.m_child_index_in_parent == 2) // Bottom left
        {
            // Get vertex index in parent tile
            uint2 l_vertex_index = uint2(l_tile_position_x >> 1, l_tile_position_y >> 1);

            l_position_0 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index, float3(0, 0, 0));
            l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(1, 0), float3(0, 0, 0));
            l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(0, 1), float3(0, 0, 0));

            if (l_vertex_index.x == 0)
            {
                l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_x_index, uint2(l_vertex_resolution_minus_one - 1, l_vertex_index.y), l_instance.m_parent_neighbour_x_offset);
            }
            else
            {
                l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x - 1, l_vertex_index.y), float3(0, 0, 0));
            }

            if (l_vertex_index.y == 0)
            {
                l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_y_index, uint2(l_vertex_index.x, l_vertex_resolution_minus_one - 1), l_instance.m_parent_neighbour_y_offset);
            }
            else
            {
                l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x, l_vertex_index.y - 1), float3(0, 0, 0));
            }
        }
        else if (l_instance.m_child_index_in_parent == 3) // Bottom right
        {
            // Get vertex index in parent tile
            uint2 l_vertex_index = uint2((l_tile_position_x + g_push_constants.m_tile_vertex_resolution) >> 1, l_tile_position_y >> 1);

            l_position_0 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index, float3(0, 0, 0));

            if (l_vertex_index.x == l_vertex_resolution_minus_one)
            {
                l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_x_index, uint2(1, l_vertex_index.y), l_instance.m_parent_neighbour_x_offset);
            }
            else
            {
                l_position_1 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(1, 0), float3(0, 0, 0));
            }

            l_position_2 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, l_vertex_index + uint2(0, 1), float3(0, 0, 0));
            l_position_3 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x - 1, l_vertex_index.y), float3(0, 0, 0));

            if (l_vertex_index.y == 0)
            {
                l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_neighbour_y_index, uint2(l_vertex_index.x, l_vertex_resolution_minus_one - 1), l_instance.m_parent_neighbour_y_offset);
            }
            else
            {
                l_position_4 = get_position(l_undisplaced_position_buffer, l_instance.m_parent_index, uint2(l_vertex_index.x, l_vertex_index.y - 1), float3(0, 0, 0));
            }
        }

        // Use cross-producs to avarage normals
        float3 l_surface_normal = 0;
        l_surface_normal += normalize(cross(l_position_2 - l_position_0, l_position_1 - l_position_0));
        l_surface_normal += normalize(cross(l_position_1 - l_position_0, l_position_4 - l_position_0));
        l_surface_normal += normalize(cross(l_position_4 - l_position_0, l_position_3 - l_position_0));
        l_surface_normal += normalize(cross(l_position_3 - l_position_0, l_position_2 - l_position_0));
        l_surface_normal = normalize(l_surface_normal);

        // Get horizontal displacement
        float3 l_horizontal_displacement = l_surface_normal;
        {
            const float l_hd_const_0 = 1.75f;
            const int l_hd_const_1 = 500;
            const int l_hd_const_2 = 5;
            const float l_hd_const_3 = -0.75f;

            // Get planet normal
            float3 l_planet_normal_01 = lerp(l_instance.m_normal_0, l_instance.m_normal_1, l_tile_position.x);
            float3 l_planet_normal_23 = lerp(l_instance.m_normal_3, l_instance.m_normal_2, l_tile_position.x);
            float3 l_planet_normal = lerp(l_planet_normal_01, l_planet_normal_23, l_tile_position.y);
            l_planet_normal = normalize(l_planet_normal);

            // Get weight by slope
            float l_hd_weight = 1.0 - saturate(dot(l_surface_normal, l_planet_normal));
            l_hd_weight = pow(l_hd_weight, l_hd_const_0);

            float2 l_fract_uv = frac(l_instance.m_tile_pos_frac_10000.xy + l_tile_position.xy * l_instance.m_tile_pos_frac_10000.zz);
            float l_hd_noise = fbm_repeat(l_fract_uv, l_hd_const_1, l_hd_const_2);

            // Get displacement size
            float l_hd_height = (l_hd_noise + l_hd_const_3) * l_hd_weight;

            l_horizontal_displacement *= l_hd_height;
        }

        l_position_ls += l_horizontal_displacement * g_push_constants.m_horizontal_displacement_scale;
    }
#endif

    float3 l_normal = sample_terrain_normal(l_instance,
                                            l_tile_position,
                                            g_push_constants.m_elevation_tile_resolution,
                                            g_push_constants.m_elevation_tile_border,
                                            l_heightmap_array);


    l_position_buffer[l_data_index] = l_position_ls;
    l_normal_buffer[l_data_index] = l_normal;
}
