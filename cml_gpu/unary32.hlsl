// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

[numthreads(1, 1, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    CML_GET_BUFFERS;

    l_tensors.Store(l_meta_data.m_tensor_offset_1 + 20, asuint(2.0f));
}
