// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // mean
// uint m_tensor_offset_2; // variance

#define GROUP_SIZE 1

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint4 l_in_shape_input;
    uint4 l_out_shape_mean;
    uint4 l_out_shape_var;

    uint l_in_rank_input = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_0)));
    uint l_out_rank_mean = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_1)));
    uint l_out_rank_var = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_2)));

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    uint l_out_byte_offset_mean = l_meta_data.m_tensor_offset_1;
    uint l_out_byte_offset_var = l_meta_data.m_tensor_offset_2;

    for (uint l_i = 0; l_i < l_in_rank_input; l_i++)
    {
        l_in_shape_input[l_i] = asuint(l_tensors.Load(uint(l_meta_data.m_tensor_offset_0 + 4 + 4 * l_i)));
    }

    for (uint l_j = 0; l_j < l_out_rank_mean; l_j++)
    {
        l_out_shape_mean[l_j] = asuint(l_tensors.Load(uint(l_meta_data.m_tensor_offset_1 + 4 + 4 * l_j)));
    }

    for (uint l_k = 0; l_k < l_out_rank_var; l_k++)
    {
        l_out_shape_var[l_k] = asuint(l_tensors.Load(uint(l_meta_data.m_tensor_offset_2 + 4 + 4 * l_k)));
    }

    l_in_byte_offset_input += 4 + 4 * l_in_rank_input;
    l_out_byte_offset_mean += 4 + 4 * l_out_rank_mean;
    l_out_byte_offset_var += 4 + 4 * l_out_rank_var;

    uint l_axis1 = asuint(l_attributes.Load(int(l_meta_data.m_attrib_offset + 4)));
    uint l_axis2 = asuint(l_attributes.Load(int(l_meta_data.m_attrib_offset + 8)));

    if (l_meta_data.m_attrib_count != 2)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }

    if (l_axis1 != 2 || l_axis2 != 3)
    {
        CML_SET_ERROR_INT(0, l_axis1);
        CML_SET_ERROR_INT(1, l_axis2);
        CML_SET_ERROR_INT(2, -99);
        CML_SET_KERNEL_ERROR;
    }

    uint l_idx_in = p_gid.x * l_in_shape_input[2] * l_in_shape_input[3];
    uint l_idx_out = p_gid.x;
    uint l_n_elements = l_in_shape_input[2] * l_in_shape_input[3];

    float l_mean = 0;

    for (l_i = 0; l_i < l_n_elements; l_i++)
    {
        l_mean += asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (l_idx_in + l_i)));
    }
    l_mean /= l_n_elements;

    float l_var = 0;

    for (l_i = 0; l_i < l_n_elements; l_i++)
    {
        float l_temp = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (l_idx_in + l_i))) - l_mean;

        l_var += l_temp * l_temp;
    }
    l_var /= l_n_elements;

    l_tensors.Store(l_out_byte_offset_mean + 4 * l_idx_out, asuint(l_mean));
    l_tensors.Store(l_out_byte_offset_var + 4 * l_idx_out, asuint(l_var));
}
