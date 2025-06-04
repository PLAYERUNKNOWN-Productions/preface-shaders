// Copyright:   PlayerUnknown Productions BV

#include "../helper_shaders/mb_common.hlsl"

[numthreads(1, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    // Reset the counter
    RWStructuredBuffer<uint> l_vt_feedback_buffer_counter = ResourceDescriptorHeap[RAYTRACING_FEEDBACK_BUFFER_COUNTER_UAV];
    l_vt_feedback_buffer_counter[0] = 0;
}
