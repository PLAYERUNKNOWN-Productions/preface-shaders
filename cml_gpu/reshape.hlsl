// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 1024
[numthreads(1024, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Every Tensor starts with its shape
    uint l_in_shape_a[6];
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    uint l_in_rank_a = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0));

    uint l_in_size_a = 1;
    for (uint l_i = 0; l_i < l_in_rank_a; l_i++)
    {
        l_in_shape_a[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 * (1 + l_i)));
        l_in_size_a *= l_in_shape_a[l_i];
    }

    l_in_byte_offset_a += 4 + 4 * l_in_rank_a;

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    uint l_out_rank_b = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_1));
    l_out_byte_offset += 4 + 4 * l_out_rank_b;

    uint l_ind0 = GROUP_SIZE * p_gid.x + p_gtid.x;

    if (l_ind0 < l_in_size_a)
    {
        float l_out = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_ind0));
        l_tensors.Store(l_out_byte_offset + 4 * l_ind0, asuint(l_out));
    }
}
