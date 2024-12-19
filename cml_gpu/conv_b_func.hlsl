// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Attributes specifically for this operation
struct conv_b_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_b_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count != 9)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif // CML_KERNEL_ERROR_HANDLING

    p_attribs.m_padding = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_stride  = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

#define l_d_o (l_out_shape[3])
#define l_d_i (l_in_shape_input[3])
#define l_out_ch (l_out_shape[1])
#define l_in_ch (l_in_shape_input[1])
#define l_d_f (l_in_shape_filter[2])
#define l_s1 (l_attribs.m_stride[0])
#define l_s2 (l_attribs.m_stride[1])
#define l_d1 (l_attribs.m_dilation[0])
#define l_d2 (l_attribs.m_dilation[1])
#define l_p1 (l_attribs.m_padding[0])
#define l_p2 (l_attribs.m_padding[2])

#define FLOAT_SIZE 4
#define GROUP_SIZE_Z 4

#if GROUP_SIZE_AT_X > 8
    #define GROUP_SIZE_X 8
    #define GROUP_SIZE_Y 8
#else
    #define GROUP_SIZE_X GROUP_SIZE_AT_X
    #define GROUP_SIZE_Y GROUP_SIZE_AT_Y
#endif

//-----------------------------------------------------------------------------
// ASSUMPTIONS
//-----------------------------------------------------------------------------
//
// l_d_f <= 4
// l_attribs.m_padding[0] = l_attribs.m_padding[1]
// l_attribs.m_padding[2] = l_attribs.m_padding[3]


groupshared float l_sum[GROUP_SIZE_AT_X * GROUP_SIZE_AT_Y * GROUP_SIZE_Z];

void conv_b_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // We cannot enable below when we have GroupMemoryBarrierWithGroupSync();
    //CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    conv_b_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint4 l_in_shape_filter;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_filter += tensor_shape(l_tensors, l_in_byte_offset_filter, l_in_shape_filter);

    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2;
    uint2 l_in_shape_bias = asuint(l_tensors.Load2(l_in_byte_offset_bias + FLOAT_SIZE));
    l_in_byte_offset_bias += FLOAT_SIZE * (1 + 2);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint l_k2 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE_AT_Y + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE_AT_X + p_gtid.x;

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        float l_value = 0.0f;

        for (uint l_k1 = p_gtid.z; l_k1 < l_in_ch; l_k1 += GROUP_SIZE_Z)
        {
            for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
            {
                int l_ind2 = l_i1 * l_s1 + l_j1 * l_d1 - l_p1;

                if (l_ind2 >= 0 && l_ind2 < l_d_i)
                {
                    uint idx_filter_row = l_j1 * l_d_f + l_k1 * l_d_f * l_d_f + l_k2 * l_d_f * l_d_f * l_in_ch;
                    float4 l_filter_row = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * idx_filter_row));

                    int l_idx_input0 = l_ind2 * l_d_i + l_k1 * l_d_i * l_d_i;

                    for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
                    {
                        int l_ind3 = l_i2 * l_s2 + l_j2 * l_d2 - l_p2;

                        if (l_ind3 >= 0 && l_ind3 < l_d_i)
                        {
                            int l_idx_input = l_ind3 + l_idx_input0;

                            float l_input = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                            float l_filter = l_filter_row[l_j2];

                            l_value += l_input * l_filter;
                        }
                    }
                }
            }
        }
        l_sum[p_gtid.z * GROUP_SIZE_AT_X * GROUP_SIZE_AT_Y + p_gtid.y * GROUP_SIZE_AT_X + p_gtid.x] = l_value;
    }

    GroupMemoryBarrierWithGroupSync();


    if (p_gtid.z == 0 && l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        float l_value = 0;

        for (uint l_i = 0; l_i < GROUP_SIZE_Z; l_i++)
        {
            l_value += l_sum[l_i * GROUP_SIZE_AT_X * GROUP_SIZE_AT_Y + p_gtid.y * GROUP_SIZE_AT_X + p_gtid.x];
        }

        // Get bias value
        // Do not add offset if shape_bias = [1, 1]
        if (l_in_shape_bias[1] > 1)
        {
            l_in_byte_offset_bias += FLOAT_SIZE * l_k2;
        }

        float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));

        uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_value + l_bias));
    }
}
