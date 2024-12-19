// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Attributes specifically for this operation
struct deconv_d_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int4 m_output_shape;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out deconv_d_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count != 13 && l_count != 9)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif // CML_KERNEL_ERROR_HANDLING

    p_attribs.m_padding = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_stride = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_output_shape = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}


#define l_d_o (l_out_shape[3])
#define l_d_i (l_in_shape_input[3])
#define l_out_ch (l_out_shape[1])
#define l_in_ch (l_in_shape_input[1])
#define l_d_f (l_in_shape_filter[2])
#define l_s1 (l_attribs.m_stride.x)
#define l_s2 (l_attribs.m_stride.y)
#define l_d1 (l_attribs.m_dilation.x)
#define l_d2 (l_attribs.m_dilation.y)
#define l_p1 (l_attribs.m_padding.x)
#define l_p2 (l_attribs.m_padding.z)

#define l_k1 (p_gid.z)
#define GROUP_SIZE_Z 2

#if (GROUP_SIZE_AT_X > 16 || GROUP_SIZE_AT_Y > 16)
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16
#else
#define GROUP_SIZE_X GROUP_SIZE_AT_X
#define GROUP_SIZE_Y GROUP_SIZE_AT_Y
#endif


groupshared float l_sum[4 * GROUP_SIZE_X * GROUP_SIZE_Y][GROUP_SIZE_Z];

void deconv_d_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    //CML_CHECK_KERNEL_ERROR;

    deconv_d_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint4 l_in_shape_filter;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_filter += tensor_shape(l_tensors, l_in_byte_offset_filter, l_in_shape_filter);

    uint4 l_in_shape_bias;
    uint l_in_rank_bias = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2));
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2 + 4 * (1 + l_in_rank_bias);

    for (uint l_i = 0; l_i < l_in_rank_bias; l_i++)
    {
        l_in_shape_bias[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2 + 4 * (1 + l_i)));
    }

    float l_bias = 0;

    if (l_in_rank_bias == 2)
    {
        if (l_in_shape_bias[1] > 1)
        {
            l_in_byte_offset_bias += 4 * l_k1;
        }

        l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));
    }

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    int l_a1 = (p_gid.y * GROUP_SIZE_Y + p_gtid.y);
    int l_a2 = (p_gid.x * GROUP_SIZE_X + p_gtid.x);
    uint l_i1 = 2 * (uint)l_a1;
    uint l_i2 = 2 * (uint)l_a2;
    uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k1 * l_d_o * l_d_o;

    float3 l_input0, l_input1, l_input2;
    float4 l_filter0, l_filter1, l_filter2, l_filter3;

    l_a1 -= 1;
    l_a2 -= 1;

    uint l_offset0 = l_a1 * l_d_i + l_a2;
    uint l_offset1 = (l_a1 + 1) * l_d_i + l_a2;
    uint l_offset2 = (l_a1 + 2) * l_d_i + l_a2;

    uint l_lin_id00 = (2 * GROUP_SIZE_X) * (2 * p_gtid.y) + (2 * p_gtid.x);
    uint l_lin_id01 = l_lin_id00 + 1;
    uint l_lin_id10 = l_lin_id00 + (2 * GROUP_SIZE_X);
    uint l_lin_id11 = l_lin_id10 + 1;

    l_sum[l_lin_id00][p_gtid.z] = 0;
    l_sum[l_lin_id01][p_gtid.z] = 0;
    l_sum[l_lin_id10][p_gtid.z] = 0;
    l_sum[l_lin_id11][p_gtid.z] = 0;

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        for (uint l_k2 = p_gtid.z; l_k2 < l_in_ch; l_k2 += GROUP_SIZE_Z)
        {
            uint l_input_offset = l_k2 * l_d_i * l_d_i;
            uint l_filter_offset = l_d_f * l_d_f * (l_k1 + l_k2 * l_out_ch);

            l_input0 = asfloat(l_tensors.Load3(l_in_byte_offset_input + 4 * (l_input_offset + l_offset0)));
            l_input1 = asfloat(l_tensors.Load3(l_in_byte_offset_input + 4 * (l_input_offset + l_offset1)));
            l_input2 = asfloat(l_tensors.Load3(l_in_byte_offset_input + 4 * (l_input_offset + l_offset2)));

            l_filter0 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_filter_offset)));
            l_filter1 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_filter_offset + 4)));
            l_filter2 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_filter_offset + 8)));
            l_filter3 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_filter_offset + 12)));

            // Padding
            if (l_a1 == -1)
            {
                l_input0 = float3(0, 0, 0);
            }
            else if (l_a1 == -2)
            {
                l_input1 = float3(0, 0, 0);
                l_input0 = float3(0, 0, 0);
            }
            else if (l_a1 == l_d_i - 2)
            {
                l_input2 = float3(0, 0, 0);
            }
            else if (l_a1 == l_d_i - 1)
            {
                l_input2 = float3(0, 0, 0);
                l_input1 = float3(0, 0, 0);
            }
            else if (l_a1 == l_d_i)
            {
                l_input0 = float3(0, 0, 0);
                l_input1 = float3(0, 0, 0);
                l_input2 = float3(0, 0, 0);
            }

            if (l_a2 == -1)
            {
                l_input0[0] = 0;
                l_input1[0] = 0;
                l_input2[0] = 0;
            }
            else if (l_a2 == -2)
            {
                l_input0[0] = 0;
                l_input1[0] = 0;
                l_input2[0] = 0;
                l_input0[1] = 0;
                l_input1[1] = 0;
                l_input2[1] = 0;
            }
            else if (l_a2 == l_d_i - 2)
            {
                l_input0[2] = 0;
                l_input1[2] = 0;
                l_input2[2] = 0;
            }
            else if (l_a2 == l_d_i - 1)
            {
                l_input0[1] = 0;
                l_input1[1] = 0;
                l_input2[1] = 0;
                l_input0[2] = 0;
                l_input1[2] = 0;
                l_input2[2] = 0;
            }
            else if (l_a2 == l_d_i)
            {
                l_input0 = float3(0, 0, 0);
                l_input1 = float3(0, 0, 0);
                l_input2 = float3(0, 0, 0);
            }

            l_sum[l_lin_id00][p_gtid.z] += l_input1[1] * l_filter1[1] + l_input1[0] * l_filter1[3] + l_input0[1] * l_filter3[1] + l_input0[0] * l_filter3[3]; // (0,0)
            l_sum[l_lin_id01][p_gtid.z] += l_input1[2] * l_filter1[0] + l_input1[1] * l_filter1[2] + l_input0[2] * l_filter3[0] + l_input0[1] * l_filter3[2]; // (0,1)
            l_sum[l_lin_id10][p_gtid.z] += l_input2[1] * l_filter0[1] + l_input2[0] * l_filter0[3] + l_input1[1] * l_filter2[1] + l_input1[0] * l_filter2[3]; // (1,0)
            l_sum[l_lin_id11][p_gtid.z] += l_input2[2] * l_filter0[0] + l_input2[1] * l_filter0[2] + l_input1[2] * l_filter2[0] + l_input1[1] * l_filter2[2]; // (1,1)
        }
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2] && p_gtid.z == 0)
    {
        l_filter0 = float4(0, 0, 0, 0);

        for (uint l_i = 0; l_i < GROUP_SIZE_Z; l_i++)
        {
            l_filter0[0] += l_sum[l_lin_id00][l_i];
            l_filter0[1] += l_sum[l_lin_id01][l_i];
            l_filter0[2] += l_sum[l_lin_id10][l_i];
            l_filter0[3] += l_sum[l_lin_id11][l_i];
        }

        l_tensors.Store(l_out_byte_offset + 4 * (l_idx_output), asuint(l_filter0[0]+ l_bias));
        l_tensors.Store(l_out_byte_offset + 4 * (l_idx_output + 1), asuint(l_filter0[1] + l_bias));
        l_tensors.Store(l_out_byte_offset + 4 * (l_idx_output + l_d_o), asuint(l_filter0[2] + l_bias));
        l_tensors.Store(l_out_byte_offset + 4 * (l_idx_output + l_d_o + 1), asuint(l_filter0[3] + l_bias));
    }
}
