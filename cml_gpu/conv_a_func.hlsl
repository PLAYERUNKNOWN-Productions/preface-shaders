// Copyright:   PlayerUnknown Productions BV

// Attributes specifically for this operation
struct conv_a_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_a_attribs_s p_attribs)
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
#endif  // CML_KERNEL_ERROR_HANDLING

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

#define GROUP_SIZE_Z 2

#if (GROUP_SIZE_AT_X > 16 || GROUP_SIZE_AT_Y > 16)
    #define GROUP_SIZE_X 16
    #define GROUP_SIZE_Y 16
#else
    #define GROUP_SIZE_X 16
    #define GROUP_SIZE_Y 16
#endif

#define l_d_o (l_out_shape[3])
#define l_d_i (l_in_shape_input[3])
#define l_out_ch (l_out_shape[1])
#define l_in_ch (l_in_shape_input[1])
#define l_d_f 3
#define l_s1 1
#define l_s2 1
#define l_d1 1
#define l_d2 1
#define l_p1 1
#define l_p2 1

#define FLOAT_SIZE 4
#define l_d_f_sqr 9

// ASSUMPTIONS
// l_d_f == 3
// l_d1 == l_d2 == l_s1 == l_s2 == l_p1 == l_p2 == 1
// l_attribs.m_padding[0] = l_attribs.m_padding[1]
// l_attribs.m_padding[2] = l_attribs.m_padding[3]
// l_out_shape[2] and l_out_shape[3] are even

groupshared float l_sum[4 * GROUP_SIZE_Y * GROUP_SIZE_X][GROUP_SIZE_Z];

void conv_a_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // We cannot enable below when we have GroupMemoryBarrierWithGroupSync();
    //CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    conv_a_attribs_s l_attribs;
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
    uint l_i1 = 2 * (p_gid.y * GROUP_SIZE_Y + p_gtid.y);
    uint l_i2 = 2 * (p_gid.x * GROUP_SIZE_X + p_gtid.x);

    // indices for top left corner
    int l_a1 = l_i1 - l_p1;
    int l_a2 = l_i2 - l_p2;

    uint l_input_offset_row0 = (uint)(l_a1 * l_d_i + l_a2);
    uint l_input_offset_row1 = (uint)((l_a1 + 1) * l_d_i + l_a2);
    uint l_input_offset_row2 = (uint)((l_a1 + 2) * l_d_i + l_a2);
    uint l_input_offset_row3 = (uint)((l_a1 + 3) * l_d_i + l_a2);

    // Each thread-group calculates 2 * GROUP_SIZE_X by 2 * GROUP_SIZE_Y
    // Each thread computes for 2x2 [2*p_gtid.x:2*p_gtid.x+2, 2*p_gtid.y:2*p_gtid.y+2]
    uint l_group_id00 = (2 * GROUP_SIZE_X) * (2 * p_gtid.y) + (2 * p_gtid.x);
    uint l_group_id01 = l_group_id00 + 1;
    uint l_group_id10 = l_group_id00 + (2 * GROUP_SIZE_X);
    uint l_group_id11 = l_group_id10 + 1;

    // Store partial results in groupshared memory for each [output_pixel][z_thread]
    l_sum[l_group_id00][p_gtid.z] = 0;
    l_sum[l_group_id01][p_gtid.z] = 0;
    l_sum[l_group_id10][p_gtid.z] = 0;
    l_sum[l_group_id11][p_gtid.z] = 0;

    uint l_d_i_sqr = l_d_i * l_d_i;
    uint l_k2_in_ch = l_k2 * l_in_ch;

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        for (uint l_k1 = p_gtid.z; l_k1 < l_in_ch; l_k1 += GROUP_SIZE_Z)
        {
            uint l_input_offset_ch = l_k1 * l_d_i_sqr;
            uint l_filter_offset_row0 = l_d_f_sqr * (l_k1 + l_k2_in_ch);

            float4 l_input_row0 = asfloat(l_tensors.Load4(l_in_byte_offset_input + FLOAT_SIZE * (l_input_offset_ch + l_input_offset_row0)));
            float4 l_input_row1 = asfloat(l_tensors.Load4(l_in_byte_offset_input + FLOAT_SIZE * (l_input_offset_ch + l_input_offset_row1)));
            float4 l_input_row2 = asfloat(l_tensors.Load4(l_in_byte_offset_input + FLOAT_SIZE * (l_input_offset_ch + l_input_offset_row2)));
            float4 l_input_row3 = asfloat(l_tensors.Load4(l_in_byte_offset_input + FLOAT_SIZE * (l_input_offset_ch + l_input_offset_row3)));

            float3 l_filter_row0 = asfloat(l_tensors.Load3(l_in_byte_offset_filter + FLOAT_SIZE * (l_filter_offset_row0)));
            float3 l_filter_row1 = asfloat(l_tensors.Load3(l_in_byte_offset_filter + FLOAT_SIZE * (l_filter_offset_row0 + l_d_f)));
            float3 l_filter_row2 = asfloat(l_tensors.Load3(l_in_byte_offset_filter + FLOAT_SIZE * (l_filter_offset_row0 + l_d_f + l_d_f)));

            // Padding
            if (l_a1 == -1)
            {
                l_input_row0 = float4(0, 0, 0, 0);
            }
            else if (l_a1 == l_d_i - 3)
            {
                l_input_row3 = float4(0, 0, 0, 0);
            }

            if (l_a2 == -1)
            {
                l_input_row0[0] = 0;
                l_input_row1[0] = 0;
                l_input_row2[0] = 0;
                l_input_row3[0] = 0;
            }
            else if (l_a2 == l_d_i - 3)
            {
                l_input_row0[3] = 0;
                l_input_row1[3] = 0;
                l_input_row2[3] = 0;
                l_input_row3[3] = 0;
            }

            // partial sums
            l_sum[l_group_id00][p_gtid.z] +=
                  l_input_row0[0] * l_filter_row0[0] + l_input_row0[1] * l_filter_row0[1] + l_input_row0[2] * l_filter_row0[2]
                + l_input_row1[0] * l_filter_row1[0] + l_input_row1[1] * l_filter_row1[1] + l_input_row1[2] * l_filter_row1[2]
                + l_input_row2[0] * l_filter_row2[0] + l_input_row2[1] * l_filter_row2[1] + l_input_row2[2] * l_filter_row2[2];
            l_sum[l_group_id01][p_gtid.z] +=
                  l_input_row0[1] * l_filter_row0[0] + l_input_row0[2] * l_filter_row0[1] + l_input_row0[3] * l_filter_row0[2]
                + l_input_row1[1] * l_filter_row1[0] + l_input_row1[2] * l_filter_row1[1] + l_input_row1[3] * l_filter_row1[2]
                + l_input_row2[1] * l_filter_row2[0] + l_input_row2[2] * l_filter_row2[1] + l_input_row2[3] * l_filter_row2[2];
            l_sum[l_group_id10][p_gtid.z] +=
                  l_input_row1[0] * l_filter_row0[0] + l_input_row1[1] * l_filter_row0[1] + l_input_row1[2] * l_filter_row0[2]
                + l_input_row2[0] * l_filter_row1[0] + l_input_row2[1] * l_filter_row1[1] + l_input_row2[2] * l_filter_row1[2]
                + l_input_row3[0] * l_filter_row2[0] + l_input_row3[1] * l_filter_row2[1] + l_input_row3[2] * l_filter_row2[2];
            l_sum[l_group_id11][p_gtid.z] +=
                  l_input_row1[1] * l_filter_row0[0] + l_input_row1[2] * l_filter_row0[1] + l_input_row1[3] * l_filter_row0[2]
                + l_input_row2[1] * l_filter_row1[0] + l_input_row2[2] * l_filter_row1[1] + l_input_row2[3] * l_filter_row1[2]
                + l_input_row3[1] * l_filter_row2[0] + l_input_row3[2] * l_filter_row2[1] + l_input_row3[3] * l_filter_row2[2];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2] && p_gtid.z == 0)
    {
        float4 l_out = float4(0, 0, 0, 0);

        for (uint l_i = 0; l_i < GROUP_SIZE_Z; l_i++)
        {
            l_out[0] += l_sum[l_group_id00][l_i];
            l_out[1] += l_sum[l_group_id01][l_i];
            l_out[2] += l_sum[l_group_id10][l_i];
            l_out[3] += l_sum[l_group_id11][l_i];
        }

        // Get bias value
        // Do not add offset if shape_bias = [1, 1]
        if (l_in_shape_bias[1] > 1)
        {
            l_in_byte_offset_bias += FLOAT_SIZE * l_k2;
        }

        float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));

        uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;

        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * (l_idx_output), asuint(l_out[0] + l_bias));
        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * (l_idx_output + 1), asuint(l_out[1] + l_bias));
        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * (l_idx_output + l_d_o), asuint(l_out[2] + l_bias));
        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * (l_idx_output + l_d_o + 1), asuint(l_out[3] + l_bias));
    }
}
