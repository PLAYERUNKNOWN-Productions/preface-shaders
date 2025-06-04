// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

#define l_dim_x l_in_shape_a
#define l_dim_y l_in_shape_b
#define l_dim_z l_out_shape

#define GROUP_SIZE 256
#define N_TENSORS 3
#define ID_OUT_TENSOR 2

[numthreads(GROUP_SIZE,1,1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    if (l_meta_data.m_attrib_count != 0)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }


    uint l_rank[N_TENSORS];
    uint l_shape[N_TENSORS][MB_CML_GPU_MAX_TENSOR_RANK];
    uint l_byte_offset_tensor[N_TENSORS] = {l_meta_data.m_tensor_offset_0,
                                            l_meta_data.m_tensor_offset_1,
                                            l_meta_data.m_tensor_offset_2};
    bool l_broadcast_x[MB_CML_GPU_MAX_TENSOR_RANK];
    bool l_broadcast_y[MB_CML_GPU_MAX_TENSOR_RANK];

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for (uint l_j = 0; l_j < l_rank[l_i]; l_j++)
        {
            l_shape[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + FLOAT_SIZE * (1 + l_j)));
        }
        for (uint l_j = l_rank[l_i]; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }
        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    uint l_n_outputs = 1;
    // number of outputs
    for (uint l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_n_outputs *= l_shape[ID_OUT_TENSOR][l_i];
    }

    for (uint l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_broadcast_x[l_i] = l_shape[0][l_i] < l_shape[ID_OUT_TENSOR][l_i];
        l_broadcast_y[l_i] = l_shape[1][l_i] < l_shape[ID_OUT_TENSOR][l_i];
    }

    uint l_idx_output = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_output < l_n_outputs)
    {
        uint l_z_coord[MB_CML_GPU_MAX_TENSOR_RANK];

        out_flatten_to_coord(l_idx_output, l_shape[ID_OUT_TENSOR], l_rank[ID_OUT_TENSOR], l_z_coord);

        uint l_idx_input_x = (uint)in_coord_to_flatten_unary(l_z_coord, l_shape[0], l_rank[0], l_broadcast_x);
        uint l_idx_input_y = (uint)in_coord_to_flatten_unary(l_z_coord, l_shape[1], l_rank[1], l_broadcast_y);

        float l_x = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_input_x));
        float l_y = asfloat(l_tensors.Load(l_byte_offset_tensor[1] + FLOAT_SIZE * l_idx_input_y));

        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_output, asuint(l_x * l_y));
    }
}
