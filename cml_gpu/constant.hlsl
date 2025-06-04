// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 1
// uint m_tensor_offset_0; // output a

#define GROUP_SIZE 32
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint4 l_out_shape;
    uint l_out_size = 1;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_0;
    uint l_out_rank = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0));

    for (uint l_i = 0; l_i < l_out_rank; l_i++)
    {
        l_out_shape[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 * (1 + l_i)));
        l_out_size *= l_out_shape[l_i];
    }
    l_out_byte_offset += 4 + 4 * l_out_rank;

    uint l_ind0 = GROUP_SIZE * GROUP_SIZE * p_gid.x + p_gi;

    if (l_ind0 < l_out_size)
    {
        float l_out = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (1 + l_out_rank + l_ind0)));
        l_tensors.Store(l_out_byte_offset + 4 * l_ind0, asuint(l_out));
    }
}
