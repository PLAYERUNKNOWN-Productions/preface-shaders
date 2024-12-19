// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
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

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for (uint l_j = 0; l_j < l_rank[l_i]; l_j++)
        {
            l_shape[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + FLOAT_SIZE * (1 + l_j)));
        }
        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    // Obtain attributes
    uint l_n_attribs = asuint(l_attributes.Load(l_meta_data.m_attrib_offset));
    uint l_axes = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + FLOAT_SIZE));
    int l_begin = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 2 * FLOAT_SIZE));
    int l_end = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 3 * FLOAT_SIZE));
    int l_stride = 1;

    if (l_begin < 0)
        l_begin += l_shape[0][l_axes];

    if (l_end <= 0)
        l_end += l_shape[0][l_axes];

    if (l_n_attribs == 4)
    {
        l_stride = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * FLOAT_SIZE));
    }

    uint l_n_outputs = 1;
    // number of outputs
    for (l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_n_outputs *= l_shape[ID_OUT_TENSOR][l_i];
    }

    uint l_idx_output = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_output < l_n_outputs)
    {
        uint l_coord[MB_CML_GPU_MAX_TENSOR_RANK];
        out_flatten_to_coord(l_idx_output, l_shape[ID_OUT_TENSOR], l_rank[ID_OUT_TENSOR], l_coord);
        l_coord[l_axes] += l_begin;

        uint l_idx_input = (uint)in_coord_to_flatten(l_coord, l_shape[0], l_rank[0]);
        float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_input));
        l_tensors.Store(l_byte_offset_tensor[1] + FLOAT_SIZE * l_idx_output, asuint(l_value));
    }
}
