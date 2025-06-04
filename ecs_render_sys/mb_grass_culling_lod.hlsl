// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"
#include "mb_grass_common.hlsl"

ConstantBuffer<cb_push_grass_culling_lod_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(MB_GRASS_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint instance_offset = dispatch_thread_id.x + g_push_constants.m_patch_instance_buffer_offset;
    StructuredBuffer<uint> item_buffer_offsets = ResourceDescriptorHeap[g_push_constants.m_patch_instance_buffer_srv];
    uint item_buffer_offset = item_buffer_offsets[instance_offset];

    StructuredBuffer<float3> tile_local_to_camera_local_offsets = ResourceDescriptorHeap[g_push_constants.m_patch_tile_to_camera_offset_buffer_srv];
    float3 tile_local_to_camera_local_offset = tile_local_to_camera_local_offsets[instance_offset];

    StructuredBuffer<sb_grass_patch_item_t> patch_items = ResourceDescriptorHeap[g_push_constants.m_patch_item_buffer_srv];
    sb_grass_patch_item_t patch_item = patch_items[item_buffer_offset];

    // Drop grass patch with invalid ID
    if (patch_item.m_type_id == GRASS_TYPE_ID_INVALID)
    {
        return;
    }

    float3 position_camera_local = patch_item.m_position_tile_local + tile_local_to_camera_local_offset;
    float distance_to_camera_squared = dot(position_camera_local, position_camera_local);
    if (distance_to_camera_squared > g_push_constants.m_culling_distance_squared)
    {
        return;
    }

    // Culling
    float4 bounding_sphere = float4(position_camera_local, patch_item.m_blade_height + RANDOM_EXTRA_HEIGHT_SCALE + RANDOM_EXTRA_PATCH_OFFSET + patch_item.m_patch_radius);
    ConstantBuffer<cb_camera_t> camera = ResourceDescriptorHeap[g_push_constants.m_camera_cbv];
    bool visible = is_sphere_inside_camera_frustum(camera, bounding_sphere);
    if (!visible)
    {
        return;
    }

    float blade_count_f = lerp(float(MAX_BLADE_COUNT), 2.0f, pow(saturate(sqrt(distance_to_camera_squared) / (GRASS_LOD_END_DISTANCE * 1.05f)), 0.75f));
    uint blade_count = ceil(blade_count_f);

    uint lod_level = 0;
    if (blade_count == GRASS_LOD_LEVEL_1_BLADE_COUNT)
    {
        lod_level = 1;
    }
    else
    {
        lod_level = 0;
    }

    RWStructuredBuffer<uint> count_buffer = ResourceDescriptorHeap[g_push_constants.m_patch_lod_count_buffer_uav];

    uint instance_local_index = 0;
    InterlockedAdd(count_buffer[lod_level], 1, instance_local_index);

    if (instance_local_index >= g_push_constants.m_patch_arguments_capacity)
    {
        uint original_value = 0;
        InterlockedExchange(count_buffer[lod_level], g_push_constants.m_patch_arguments_capacity, original_value);
        return;
    }

    RWStructuredBuffer<sb_grass_patch_argument_t> patch_arguments = ResourceDescriptorHeap[NonUniformResourceIndex(g_push_constants.m_patch_arguments_uavs[lod_level])];
    patch_arguments[instance_local_index].m_position_camera_local = position_camera_local;
    patch_arguments[instance_local_index].m_ground_normal = patch_item.m_ground_normal;
    patch_arguments[instance_local_index].m_color = patch_item.m_color;
    patch_arguments[instance_local_index].m_ground_color = patch_item.m_ground_color;
    patch_arguments[instance_local_index].m_random_seed = patch_item.m_random_seed;
    patch_arguments[instance_local_index].m_blade_width = patch_item.m_blade_width;
    patch_arguments[instance_local_index].m_blade_height = patch_item.m_blade_height;
    patch_arguments[instance_local_index].m_patch_radius = patch_item.m_patch_radius;
    patch_arguments[instance_local_index].m_blade_count = blade_count;
}
