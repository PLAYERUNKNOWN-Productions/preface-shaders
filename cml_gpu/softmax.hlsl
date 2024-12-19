// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0;
// uint m_tensor_offset_1;

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
#define GROUP_SIZE 64
#define N_TENSORS 2
#define ID_OUT_TENSOR 1

int flatten_input_to_global(in uint p_in_flat, in uint p_out_flat, in uint p_shape[MB_CML_GPU_MAX_TENSOR_RANK],
    in uint p_axes[MB_CML_GPU_MAX_TENSOR_RANK], in uint p_rank)
{
    uint l_temp1 = p_in_flat;
    uint l_temp2 = p_out_flat;

    uint l_ind[MB_CML_GPU_MAX_TENSOR_RANK] = { 0,0,0,0,0,0,0,0 };

    // a) initially put the output coordinate
    for (int l_i = p_rank - 1; l_i >= 0; l_i--)
    {
        l_ind[l_i] = l_temp2 % p_shape[l_i];
        l_temp2 /= p_shape[l_i];
    }

    // b) change the indices that will be iterated
    for (l_i = p_rank - 1; l_i >= 0; l_i--)
    {
        if (p_axes[l_i] == 1) {
            l_ind[l_i] = l_temp1 % p_shape[l_i];
            l_temp1 /= p_shape[l_i];
        }
    }

    // c) get a flattened index from coordinate b)
    uint l_flat_idx = l_ind[0];
    for (l_i = 1; l_i < MB_CML_GPU_MAX_TENSOR_RANK; l_i++)
    {
        l_flat_idx = l_flat_idx * p_shape[l_i] + l_ind[l_i];
    }
    return l_flat_idx;
}


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
        for (l_j = l_rank[l_i]; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }

        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    // Retrieve reduction axes
    uint l_axes[MB_CML_GPU_MAX_TENSOR_RANK] = { 0,0,0,0,0,0,0,0 };
    uint l_axis_dim;
    for (l_i = 0; l_i < l_meta_data.m_attrib_count; l_i++)
    {
        l_axis_dim = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + FLOAT_SIZE * (1 + l_i)));
        l_axes[l_axis_dim] = 1;
    }

    uint l_n_outputs = get_shape_size(l_shape[1], l_rank[1]);
    uint l_n_inputs = get_shape_size(l_shape[0], l_rank[0]);
    uint l_n_summed_elements = l_shape[0][l_axis_dim];
    uint l_idx_out = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_out < l_n_outputs)
    {
        float l_max = -9999999999999.0;
        float l_value = 0.0;
        float l_sum = 0.0;
        float l_answer = 0.0;

        // max reduce
        for (l_i = 0; l_i < l_n_summed_elements; l_i++)
        {
            uint l_idx_in = flatten_input_to_global(l_i, l_idx_out, l_shape[0], l_axes, l_rank[0]);
            l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_in));

            if (l_value > l_max)
            {
                l_max = l_value;
            }
        }

        // sum reduce of exp(x-m)
        for (l_i = 0; l_i < l_n_summed_elements; l_i++)
        {
            uint l_idx_in = flatten_input_to_global(l_i, l_idx_out, l_shape[0], l_axes, l_rank[0]);
            l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_in));
            l_sum += exp(l_value - l_max);
        }

        l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_out));

        // softmax
        l_answer = exp(l_value - l_max) / l_sum;

        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_out, asuint(l_answer));
    }
}
