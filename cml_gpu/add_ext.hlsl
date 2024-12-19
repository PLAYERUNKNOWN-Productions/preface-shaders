// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

DEF_THREAD_GROUP_SIZE_BINARY
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

    uint l_out_rank = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_2)));
    if (l_out_rank == 5)
    {
        // Every Tensor starts with its shape
        uint l_in_shape_a[6];
        uint l_in_byte_offset_a = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_0, l_in_shape_a);

        uint l_in_shape_b[6];
        uint l_in_byte_offset_b = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_1, l_in_shape_b);

        uint l_out_shape[6];
        uint l_out_byte_offset = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_2, l_out_shape);

        uint l_nbroadcast_x[6];
        uint l_nbroadcast_y[6];
        for (int l_i = 0; l_i < 6; l_i++)
        {
            l_nbroadcast_x[l_i] = 1 - (l_in_shape_a[l_i] < l_out_shape[l_i]);
            l_nbroadcast_y[l_i] = 1 - (l_in_shape_b[l_i] < l_out_shape[l_i]);
        }

        uint l_id0 = p_gid.z / l_out_shape[1] / l_out_shape[2];
        uint l_remain = p_gid.z % (l_out_shape[1] * l_out_shape[2]);
        uint l_id1 = l_remain / l_out_shape[2];
        uint l_id2 = l_remain % l_out_shape[2];
        uint l_id3 = p_gid.y * GROUP_SIZE_BINARY_Y + p_gtid.y;
        uint l_id4 = p_gid.x * GROUP_SIZE_BINARY_X + p_gtid.x;

        if (l_id3 < l_out_shape[3] && l_id4 < l_out_shape[4])
        {
            uint l_idx_z = l_id0 * l_out_shape[1] * l_out_shape[2] * l_out_shape[3] * l_out_shape[4]
                         + l_id1 * l_out_shape[2] * l_out_shape[3] * l_out_shape[4]
                         + l_id2 * l_out_shape[3] * l_out_shape[4]
                         + l_id3 * l_out_shape[4]
                         + l_id4;

            uint l_idx_x = l_id0 * l_nbroadcast_x[0] * l_in_shape_a[1] * l_in_shape_a[2] * l_in_shape_a[3] * l_in_shape_a[4]
                         + l_id1 * l_nbroadcast_x[1] * l_in_shape_a[2] * l_in_shape_a[3] * l_in_shape_a[4]
                         + l_id2 * l_nbroadcast_x[2] * l_in_shape_a[3] * l_in_shape_a[4]
                         + l_id3 * l_nbroadcast_x[3] * l_in_shape_a[4]
                         + l_id4 * l_nbroadcast_x[4];

            uint l_idx_y = l_id0 * l_nbroadcast_y[0] * l_in_shape_b[1] * l_in_shape_b[2] * l_in_shape_b[3] * l_in_shape_b[4]
                         + l_id1 * l_nbroadcast_y[1] * l_in_shape_b[2] * l_in_shape_b[3] * l_in_shape_b[4]
                         + l_id2 * l_nbroadcast_y[2] * l_in_shape_b[3] * l_in_shape_b[4]
                         + l_id3 * l_nbroadcast_y[3] * l_in_shape_b[4]
                         + l_id4 * l_nbroadcast_y[4];

            float l_x = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_idx_x));
            float l_y = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_idx_y));
            float l_z = l_x + l_y;

            // Store result
            l_tensors.Store(l_out_byte_offset + 4 * l_idx_z, asuint(l_z));
        }
    }
    else if (l_out_rank == 6)
    {
        // Every Tensor starts with its shape
        uint l_in_shape_a[6];
        uint l_in_byte_offset_a = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_0, l_in_shape_a);

        uint l_in_shape_b[6];
        uint l_in_byte_offset_b = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_1, l_in_shape_b);

        uint l_out_shape[6];
        uint l_out_byte_offset = tensor_shape(l_tensors, l_meta_data.m_tensor_offset_2, l_out_shape);

        uint l_nbroadcast_x[6];
        uint l_nbroadcast_y[6];
        for (int l_i = 0; l_i < 6; l_i++)
        {
            l_nbroadcast_x[l_i] = 1 - (l_in_shape_a[l_i] < l_out_shape[l_i]);
            l_nbroadcast_y[l_i] = 1 - (l_in_shape_b[l_i] < l_out_shape[l_i]);
        }

        uint l_out_shape_23 = l_out_shape[2] * l_out_shape[3];
        uint l_out_shape_123 = l_out_shape[1] * l_out_shape_23;
        uint l_id0 = p_gid.z / l_out_shape_123;
        uint l_remain = p_gid.z % l_out_shape_123;
        uint l_id1 = l_remain / l_out_shape_23;
        l_remain = l_remain % l_out_shape_23;
        uint l_id2 = l_remain / l_out_shape[3];
        uint l_id3 = l_remain % l_out_shape[3];
        uint l_id4 = p_gid.y * GROUP_SIZE_BINARY_Y + p_gtid.y;
        uint l_id5 = p_gid.x * GROUP_SIZE_BINARY_X + p_gtid.x;

        if (l_id4 < l_out_shape[4] && l_id5 < l_out_shape[5])
        {
            uint l_idx_z = l_id0 * l_out_shape[1] * l_out_shape[2] * l_out_shape[3] * l_out_shape[4] * l_out_shape[5]
                         + l_id1 * l_out_shape[2] * l_out_shape[3] * l_out_shape[4] * l_out_shape[5]
                         + l_id2 * l_out_shape[3] * l_out_shape[4] * l_out_shape[5]
                         + l_id3 * l_out_shape[4] * l_out_shape[5]
                         + l_id4 * l_out_shape[5]
                         + l_id5;

            uint l_idx_x = l_id0 * l_nbroadcast_x[0] * l_in_shape_a[1] * l_in_shape_a[2] * l_in_shape_a[3] * l_in_shape_a[4] * l_in_shape_a[5]
                         + l_id1 * l_nbroadcast_x[1] * l_in_shape_a[2] * l_in_shape_a[3] * l_in_shape_a[4] * l_in_shape_a[5]
                         + l_id2 * l_nbroadcast_x[2] * l_in_shape_a[3] * l_in_shape_a[4] * l_in_shape_a[5]
                         + l_id3 * l_nbroadcast_x[3] * l_in_shape_a[4] * l_in_shape_a[5]
                         + l_id4 * l_nbroadcast_x[4] * l_in_shape_a[5]
                         + l_id5 * l_nbroadcast_x[5];

            uint l_idx_y = l_id0 * l_nbroadcast_y[0] * l_in_shape_b[1] * l_in_shape_b[2] * l_in_shape_b[3] * l_in_shape_b[4] * l_in_shape_b[5]
                         + l_id1 * l_nbroadcast_y[1] * l_in_shape_b[2] * l_in_shape_b[3] * l_in_shape_b[4] * l_in_shape_b[5]
                         + l_id2 * l_nbroadcast_y[2] * l_in_shape_b[3] * l_in_shape_b[4] * l_in_shape_b[5]
                         + l_id3 * l_nbroadcast_y[3] * l_in_shape_b[4] * l_in_shape_b[5]
                         + l_id4 * l_nbroadcast_y[4] * l_in_shape_b[5]
                         + l_id5 * l_nbroadcast_y[5];

            float l_x = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_idx_x));
            float l_y = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_idx_y));
            float l_z = l_x + l_y;

            // Store result
            l_tensors.Store(l_out_byte_offset + 4 * l_idx_z, asuint(l_z));
        }
    }
}
