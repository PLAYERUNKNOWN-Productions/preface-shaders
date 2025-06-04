// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 256
#define MAXDIM 6
[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_rank;
    uint l_dim_input[MAXDIM];
    uint l_dim_output[MAXDIM];
    uint l_axes[MAXDIM];

    l_rank = l_tensors.Load(l_meta_data.m_tensor_offset_0);

    uint l_n_output = 1;

    for (uint l_i = l_rank; l_i < MAXDIM; l_i++)
    {
        l_dim_input[l_i] = 1;
        l_dim_output[l_i] = 1;
        l_axes[l_i] = l_i;
    }

    for (uint l_j = 0; l_j < l_rank; l_j++)
    {
        l_dim_input[l_j] = l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 + 4 * l_j);
        l_dim_output[l_j] = l_tensors.Load(l_meta_data.m_tensor_offset_1 + 4 + 4 * l_j);
        l_axes[l_j] = l_attributes.Load(l_meta_data.m_attrib_offset + 4 + 4 * l_j);
        l_n_output *= l_dim_output[l_j];
    }

    uint l_tensor_in_offset = l_meta_data.m_tensor_offset_0 + 4 + 4 * l_rank;
    uint l_tensor_out_offset = l_meta_data.m_tensor_offset_1 + 4 + 4 * l_rank;

    uint l_ind0 = GROUP_SIZE * p_gid.x + p_gi;

    if (l_ind0 < l_n_output)
    {
        uint l_ind_out[MAXDIM];
        uint l_ind = l_ind0;

        uint l_temp;

        l_temp = l_ind / l_dim_input[5];
        l_ind_out[5] = l_ind - l_temp * l_dim_input[5];
        l_ind = l_temp;

        l_temp = l_ind / l_dim_input[4];
        l_ind_out[4] = l_ind - l_temp * l_dim_input[4];
        l_ind = l_temp;

        l_temp = l_ind / l_dim_input[3];
        l_ind_out[3] = l_ind - l_temp * l_dim_input[3];
        l_ind = l_temp;

        l_temp = l_ind / l_dim_input[2];
        l_ind_out[2] = l_ind - l_temp * l_dim_input[2];
        l_ind = l_temp;

        l_temp = l_ind / l_dim_input[1];
        l_ind_out[1] = l_ind - l_temp * l_dim_input[1];
        l_ind = l_temp;

        l_ind_out[0] = l_ind;

        uint l_ind_out1 = l_ind_out[l_axes[5]]
                        + l_dim_output[5] * (l_ind_out[l_axes[4]] + l_dim_output[4] * (l_ind_out[l_axes[3]]
                                           + l_dim_output[3] * (l_ind_out[l_axes[2]]
                                                              + l_dim_output[2] * (l_ind_out[l_axes[1]]
                                                                                 + l_dim_output[1] * l_ind_out[l_axes[0]]))));

        float l_out = asfloat(l_tensors.Load(l_tensor_in_offset + 4 * l_ind0));
        l_tensors.Store(l_tensor_out_offset + 4 * l_ind_out1, asuint(l_out));
    }
}
