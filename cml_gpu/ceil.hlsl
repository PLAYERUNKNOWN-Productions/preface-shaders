// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

DEF_THREAD_GROUP_SIZE_UNARY
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Get input
    uint l_out_byte_offset;
    float l_a_in;

    if (prepare_unary_operation(l_meta_data, l_tensors, p_dispatch_thread_id, l_out_byte_offset, l_a_in))
    {
        // Perform operation
        float l_out = ceil(l_a_in);

        // Store result
        l_tensors.Store(l_out_byte_offset, asuint(l_out));
    }
}
