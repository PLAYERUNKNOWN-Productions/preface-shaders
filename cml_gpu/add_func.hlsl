// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

float add(in float p_a, in float p_b)
{
    return (p_a + p_b);
}

#define GROUP_SIZE 32

void add_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    if (l_meta_data.m_attrib_count != 0)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }

    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_rank_a = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_0)));
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_in_shape_b;
    uint l_in_rank_b = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_1)));
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_b += tensor_shape(l_tensors, l_in_byte_offset_b, l_in_shape_b);

    uint4 l_out_shape;
    uint l_in_rank_out = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_2)));
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_2;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);


#define l_dim_x l_in_shape_a
#define l_dim_y l_in_shape_b
#define l_dim_z l_out_shape
    uint4 l_nbroadcast_x;
    uint4 l_nbroadcast_y;

    // Match dimensions of x and y with z's
    for (int l_i = 0; l_i < 4; l_i++)
    {
        // Check if broadcasting is needed on any dimensions
        l_nbroadcast_x[l_i] = 1 - (l_in_rank_a <= l_i || l_in_shape_a[l_i] < l_dim_z[l_i]);
        l_nbroadcast_y[l_i] = 1 - (l_in_rank_b <= l_i || l_in_shape_b[l_i] < l_dim_z[l_i]);

        if (l_nbroadcast_x[l_i] == 0)
        {
            l_dim_x[l_i] = 1;
        }

        if (l_nbroadcast_y[l_i] == 0)
        {
            l_dim_y[l_i] = 1;
        }
    }

    uint l_id0 = (uint)(p_gid.z / l_out_shape[1]);
    uint l_id1 = p_gid.z % l_out_shape[1];

    uint l_idx_z0 = l_id0 * (l_dim_z[1] * l_dim_z[2] * l_dim_z[3])
                  + l_id1 * (l_dim_z[2] * l_dim_z[3]);

    uint l_idx_x0 = l_id0 * l_nbroadcast_x[0] * (l_dim_x[1] * l_dim_x[2] * l_dim_x[3])
                  + l_id1 * l_nbroadcast_x[1] * (l_dim_x[2] * l_dim_x[3]);

    uint l_idx_y0 = l_id0 * l_nbroadcast_y[0] * (l_dim_y[1] * l_dim_y[2] * l_dim_y[3])
                  + l_id1 * l_nbroadcast_y[1] * (l_dim_y[2] * l_dim_y[3]);

    for (uint l_iii = 0; l_iii < 2; l_iii++)
    {
        for (uint l_jjj = 0; l_jjj < 2; l_jjj++)
        {
            uint l_id2 = 2 * p_gid.y * GROUP_SIZE + p_gtid.y + l_iii * GROUP_SIZE;
            uint l_id3 = 2 * p_gid.x * GROUP_SIZE + p_gtid.x + l_jjj * GROUP_SIZE;

            if (l_id2 < l_out_shape[2] && l_id3 < l_out_shape[3])
            {
                uint l_idx_z = l_idx_z0 + l_id2 * l_dim_z[3] + l_id3;
                uint l_idx_x = l_idx_x0 + l_id2 * l_nbroadcast_x[2] * l_dim_x[3] + l_id3 * l_nbroadcast_x[3];
                uint l_idx_y = l_idx_y0 + l_id2 * l_nbroadcast_y[2] * l_dim_y[3] + l_id3 * l_nbroadcast_y[3];

                float l_x = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_idx_x));
                float l_y = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_idx_y));
                float l_z = l_x + l_y;

                // Store result
                l_tensors.Store(l_out_byte_offset + 4 * l_idx_z, asuint(l_z));
            }
        }
    }
}
