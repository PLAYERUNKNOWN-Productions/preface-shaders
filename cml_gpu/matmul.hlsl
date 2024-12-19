// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint4 l_in_shape_a, l_in_shape_b, l_out_shape_C;

    // Get ranks
    uint l_in_rank_a = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0));
    uint l_in_rank_b = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_1));    
    uint l_out_rank_C = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2));

    // Get shapes
    for (uint l_i = 0; l_i < l_in_rank_a; l_i++)
    {
        l_in_shape_a[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 + 4 * l_i));
    }
    for (uint l_j = 0; l_j < l_in_rank_b; l_j++)
    {
        l_in_shape_b[l_j] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_1 + 4 + 4 * l_j));
    }
    for (uint l_k = 0; l_k < l_out_rank_C; l_k++)
    {
        l_out_shape_C[l_k] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2 + 4 + 4 * l_k));
    }

    // Get location of data
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0 + 4 * (1 + l_in_rank_a);
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1 + 4 * (1 + l_in_rank_b);
    uint l_out_byte_offset_C = l_meta_data.m_tensor_offset_2 + 4 * (1 + l_out_rank_C);

    // Are one or both matices transposed?
    uint l_tr_a = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4));
    uint l_tr_b = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 8));

    // dimension of row and columns
    uint row = l_in_rank_a - 2;
    uint l_col = l_in_rank_a - 1;
    uint l_iter = (1 - l_tr_a) * l_col + l_tr_a * row;  // for the for-loop

    uint l_id1 = p_gid.z;
    uint l_id2 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE + p_gtid.x;
    

    if (l_id2 < l_out_shape_C[row] && l_id3 < l_out_shape_C[l_col])
    {
        //uint l_id1_a = (l_in_rank_a == 4 && l_in_shape_a[1] > 1) ? l_id1 : 0;
        //uint l_id1_b = (l_in_rank_b == 4 && l_in_shape_b[1] > 1) ? l_id1 : 0;
        uint l_id1_a = (l_in_rank_a > 2 && l_in_shape_a[1] > 1) ? l_id1 : 0;
        uint l_id1_b = (l_in_rank_b > 2 && l_in_shape_b[1] > 1) ? l_id1 : 0;

        uint l_loc_a = l_id1_a * l_in_shape_a[row] * l_in_shape_a[l_col];
        uint l_loc_b = l_id1_b * l_in_shape_b[row] * l_in_shape_b[l_col];

        float l_sum = 0;

        for (uint l_i = 0; l_i < l_in_shape_a[l_iter]; l_i++)
        {
            uint l_addr_a = l_loc_a + (1 - l_tr_a) * (l_id2 * l_in_shape_a[l_col] + l_i) + l_tr_a * (l_id2 + l_in_shape_a[l_col] * l_i);
            uint l_addr_b = l_loc_b + (1 - l_tr_b) * (l_i * l_in_shape_b[l_col] + l_id3) + l_tr_b * (l_i + l_id3 * l_in_shape_b[l_col]);

            float l_element_a = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_addr_a));
            float l_element_b = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_addr_b));

            l_sum += l_element_a * l_element_b;
        }

        // Store result
        l_tensors.Store(l_out_byte_offset_C + 4 * (l_id1 * l_out_shape_C[l_col] * l_out_shape_C[row] + l_id2 * l_out_shape_C[l_col] + l_id3), asuint(l_sum));
    }
}
