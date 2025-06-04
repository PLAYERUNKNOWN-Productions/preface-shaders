// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Input c
// uint m_tensor_offset_3; // Output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_in_shape_a[2], l_in_shape_b[2], l_in_shape_c[2], l_out_shape[2];

    // Get ranks
    uint l_in_rank_a = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0));
    uint l_in_rank_b = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_1));
    uint l_in_rank_c = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2));
    uint l_out_rank = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_3));

    // Get shapes
    for (uint l_i = 0; l_i < l_in_rank_a; l_i++)
    {
        l_in_shape_a[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 + 4 * l_i));
    }
    for (uint l_j = 0; l_j < l_in_rank_b; l_j++)
    {
        l_in_shape_b[l_j] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_1 + 4 + 4 * l_j));
    }
    for (uint l_k = 0; l_k < l_in_rank_c; l_k++)
    {
        l_in_shape_c[l_k] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2 + 4 + 4 * l_k));
    }
    for (uint l_l = 0; l_l < l_out_rank; l_l++)
    {
        l_out_shape[l_l] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_3 + 4 + 4 * l_l));
    }

    // Get location of data
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0 + 4 * (1 + l_in_rank_a);
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1 + 4 * (1 + l_in_rank_b);
    uint l_in_byte_offset_c = l_meta_data.m_tensor_offset_2 + 4 * (1 + l_in_rank_c);
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 4 * (1 + l_out_rank);

    // Are one or both matices transposed?
    uint l_tr_a = 0;
    uint l_tr_b = 1;

    // Dimension of row and columns
    uint l_row = l_in_rank_a - 2;
    uint l_col = l_in_rank_a - 1;
    uint l_iter = (1 - l_tr_a) * l_col + l_tr_a * l_row;  // for the for-loop

    uint l_id1 = p_gid.z;
    uint l_id2 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_id2 < l_out_shape[l_row] && l_id3 < l_out_shape[l_col])
    {
        uint l_id1_a = (l_in_rank_a > 2 && l_in_shape_a[1] > 1) ? l_id1 : 0;
        uint l_id1_b = (l_in_rank_b > 2 && l_in_shape_b[1] > 1) ? l_id1 : 0;

        uint l_loc_a = l_id1_a * l_in_shape_a[l_row] * l_in_shape_a[l_col];
        uint l_loc_b = l_id1_b * l_in_shape_b[l_row] * l_in_shape_b[l_col];

        float l_sum = 0;

        for (uint l_i = 0; l_i < l_in_shape_a[l_iter]; l_i++)
        {
            uint l_addr_a = l_loc_a + (1 - l_tr_a) * (l_id2 * l_in_shape_a[l_col] + l_i) + l_tr_a * (l_id2 + l_in_shape_a[l_col] * l_i);
            uint l_addr_b = l_loc_b + (1 - l_tr_b) * (l_i * l_in_shape_b[l_col] + l_id3) + l_tr_b * (l_i + l_id3 * l_in_shape_b[l_col]);

            float l_element_a = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_addr_a));
            float l_element_b = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_addr_b));

            l_sum += l_element_a * l_element_b;
        }

        uint l_addr_c = l_id1 * l_out_shape[l_col] * l_out_shape[l_row] + l_id2 * l_out_shape[l_col] + l_id3;
        l_sum += asfloat(l_tensors.Load(l_in_byte_offset_c + 4 * l_addr_c));

        // Store result
        l_tensors.Store(l_out_byte_offset + 4 * l_addr_c, asuint(l_sum));
    }
}
