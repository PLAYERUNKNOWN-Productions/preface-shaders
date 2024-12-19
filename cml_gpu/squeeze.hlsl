// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 1024
#define N_TENSORS 2
#define ID_OUT_TENSOR 1

[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint    l_rank[N_TENSORS];
    uint    l_shape[N_TENSORS][MB_CML_GPU_MAX_TENSOR_RANK];
    uint    l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0,
                                                l_meta_data.m_tensor_offset_1 };
    uint    l_n_outputs = 1;

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for (uint l_j = 0; l_j < l_rank[l_i]; l_j++)
        {
            l_shape[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + FLOAT_SIZE * (1 + l_j)));
        }
        for (l_j = l_rank[l_i]; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }

        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    // number of outputs
    for (l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_n_outputs *= l_shape[ID_OUT_TENSOR][l_i];
    }

    uint l_idx_out = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_out < l_n_outputs)
    {
        float l_a = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_out));
        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_out, asuint(l_a));
    }
}
