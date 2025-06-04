// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

ConstantBuffer<cb_push_grass_lod_reset_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

[numthreads(MB_GRASS_THREADGROUP_SIZE, 1, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x >= g_push_constants.m_num_lod_levels)
    {
        return;
    }

    RWStructuredBuffer<uint> count_buffer = ResourceDescriptorHeap[g_push_constants.m_count_buffer_uav];
    count_buffer[dispatch_thread_id.x] = 0;
}
