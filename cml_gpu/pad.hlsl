// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 1024
#define N_TENSORS 2
#define ID_OUT_TENSOR 1


int in_coord_to_flatten_pad(in int in_coord[MB_CML_GPU_MAX_TENSOR_RANK], in uint in_shape[MB_CML_GPU_MAX_TENSOR_RANK], in uint rank)
{
    int flat_idx = in_coord[0];

    for (uint i = 1; i < rank; i++)
    {
        flat_idx = flat_idx * in_shape[i] + in_coord[i];
    }

    return flat_idx;
}

[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    //if (l_meta_data.m_attrib_count != 9)
    //{
    //    CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
    //    CML_SET_KERNEL_ERROR;
    //}

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
            l_shape[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + INT_SIZE * (1 + l_j)));
        }
        for (uint l_j = l_rank[l_i]; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }

        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    // number of outputs
    for (uint l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_n_outputs *= l_shape[ID_OUT_TENSOR][l_i];
    }

    // Retrieve padding
    uint l_padding[2 * MB_CML_GPU_MAX_TENSOR_RANK] = { 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 };

    for (uint l_i = 0; l_i < l_meta_data.m_attrib_count - 1; l_i++)
    {
        l_padding[l_i] = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + INT_SIZE * (1 + l_i)));
    }


    uint l_idx_out = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_out < l_n_outputs)
    {
        uint l_output_idx[MB_CML_GPU_MAX_TENSOR_RANK];
        int l_input_idx[MB_CML_GPU_MAX_TENSOR_RANK];
        uint l_end_idx[MB_CML_GPU_MAX_TENSOR_RANK];

        // e
        for (uint l_i = 0; l_i < l_rank[0]; l_i++)
        {
            l_end_idx[l_i] = l_shape[0][l_i];
            if (0 > l_padding[2 * l_i + 1])
                l_end_idx[l_i] += l_padding[2 * l_i + 1];
        }

        // i
        out_flatten_to_coord(l_idx_out, l_shape[ID_OUT_TENSOR], l_rank[ID_OUT_TENSOR], l_output_idx);

        // s
        bool l_cond = true;
        for (uint l_i = 0; l_i < l_rank[0]; l_i++)
        {
            l_input_idx[l_i] = l_output_idx[l_i] - l_padding[2 * l_i];
            l_cond = l_cond && (0 <= l_input_idx[l_i]);
            l_cond = l_cond && (l_input_idx[l_i] < l_end_idx[l_i]);
        }

        float l_value = 0.0;
        if (l_cond)
        {
            uint l_idx_in = (uint)in_coord_to_flatten_pad(l_input_idx, l_shape[0], l_rank[0]);
            l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_in));
        }

        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_out, asuint(l_value));
    }
}
