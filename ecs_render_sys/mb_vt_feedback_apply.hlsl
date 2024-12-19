// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "../helper_shaders/mb_common.hlsl"

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

// CBV
ConstantBuffer<cb_push_vt_feedback_apply_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------

void write_sampler_feedback(sb_vt_feedback_t l_vt_feedback,
                            uint p_texture_srv,
                            uint p_sampler_feedback_uav)
{
    // Early exit if texture is not available
    if (p_texture_srv == RAL_NULL_BINDLESS_INDEX ||
        p_sampler_feedback_uav == RAL_NULL_BINDLESS_INDEX)
    {
        return;
    }

    // Resolve resources
    Texture2D l_texture = ResourceDescriptorHeap[p_texture_srv];
    FeedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED> l_feedback_texture = ResourceDescriptorHeap[p_sampler_feedback_uav];

    // Write feedback
    l_feedback_texture.WriteSamplerFeedbackLevel(l_texture, (SamplerState)SamplerDescriptorHeap[SAMPLER_LINEAR_WRAP], l_vt_feedback.m_uv, l_vt_feedback.m_min_mip_level);
}

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(RAYTRACING_VT_FEEDBACK_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Skip items that are outside of buffer bounds
    if (p_dispatch_thread_id.x >= g_push_constants.m_vt_feedback_buffer_capacity)
    {
        return;
    }

    // Skip uninitialized items
    StructuredBuffer<uint> l_vt_feedback_buffer_counter = ResourceDescriptorHeap[RAYTRACING_FEEDBACK_BUFFER_COUNTER_SRV];
    if (p_dispatch_thread_id.x >= l_vt_feedback_buffer_counter[0])
    {
        return;
    }

    // Get feedback entry
    StructuredBuffer<sb_vt_feedback_t> l_vt_feedback_buffer = ResourceDescriptorHeap[RAYTRACING_FEEDBACK_BUFFER_SRV];
    sb_vt_feedback_t l_vt_feedback = l_vt_feedback_buffer[p_dispatch_thread_id.x];

    // Get render item
    StructuredBuffer<sb_render_item_t> l_render_items_buffer = ResourceDescriptorHeap[g_push_constants.m_render_item_buffer_srv];
    sb_render_item_t l_render_item = l_render_items_buffer[l_vt_feedback.m_render_item_id];

    // Get material
    StructuredBuffer<sb_geometry_pbr_material_t> l_pbr_material_list = ResourceDescriptorHeap[NonUniformResourceIndex(l_render_item.m_material_buffer_srv)];
    sb_geometry_pbr_material_t l_pbr_material = l_pbr_material_list[l_render_item.m_material_index];

    // Write sampler feedback for all material's textures
    write_sampler_feedback(l_vt_feedback, NonUniformResourceIndex(l_pbr_material.m_base_color_texture_srv), NonUniformResourceIndex(l_pbr_material.m_base_color_sampler_feedback_uav));
    write_sampler_feedback(l_vt_feedback, NonUniformResourceIndex(l_pbr_material.m_metallic_roughness_texture_srv), NonUniformResourceIndex(l_pbr_material.m_metallic_roughness_sampler_feedback_uav));
    write_sampler_feedback(l_vt_feedback, NonUniformResourceIndex(l_pbr_material.m_normal_map_texture_srv), NonUniformResourceIndex(l_pbr_material.m_normal_map_sampler_feedback_uav));
}
