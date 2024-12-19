// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Attributes specifically for this operation
struct deconv_b_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int4 m_output_shape;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out deconv_b_attribs_s p_attribs)
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
    p_attribs.m_stride  = p_attrib_buffer.Load2(l_byte_offset);
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
#define l_i1 (p_gid.y * GROUP_SIZE_AT_Y + p_gtid.y)
#define l_i2 (p_gid.x * GROUP_SIZE_AT_X + p_gtid.x)
#define l_idx_output (l_i2 + l_i1 * l_d_o + l_k1 * l_d_o * l_d_o)

void deconv_b_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    deconv_b_attribs_s l_attribs;
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

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    float l_temp = 0.0f;

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        for (uint l_k2 = 0; l_k2 < l_in_ch; l_k2++)
        {
            for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
            {
                uint l_idx_filter = l_j1 * l_d_f + l_k1 * l_d_f * l_d_f + l_k2 * l_d_f * l_d_f * l_out_ch;
                float4 l_filter1 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * l_idx_filter));

                for (uint l_j2 = 0; l_j2 < 4; l_j2++)
                {
                    int l_a_ind2 = l_i1 + l_p1 - l_j1 * l_d1;
                    int l_a_ind3 = l_i2 + l_p2 - l_j2 * l_d2;

                    int l_ind2 = l_a_ind2 / l_s1;
                    int l_ind3 = l_a_ind3 / l_s2;

                    l_a_ind2 -= l_ind2 * l_s1;
                    l_a_ind3 -= l_ind3 * l_s2;

                    int l_idx_input = l_ind3 + l_ind2 * l_d_i + l_k2 * l_d_i * l_d_i;

                    if (l_a_ind2 == 0 && l_a_ind3 == 0)
                    {
                        if (l_ind2 >= 0 && l_ind3 >= 0 && l_ind2 < l_d_i && l_ind3 < l_d_i)
                        {                        
                            float l_t1 = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                            float l_t2 = l_filter1[l_j2];

                            l_temp += l_t1 * l_t2;
                        }
                    }
                }

                l_filter1 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_idx_filter + 4)));

                for (uint l_j3 = 4; l_j3 < l_d_f; l_j3++)
                {
                    int l_a_ind2 = l_i1 + l_p1 - l_j1 * l_d1;
                    int l_a_ind3 = l_i2 + l_p2 - l_j3 * l_d2;

                    int l_ind2 = l_a_ind2 / l_s1;
                    int l_ind3 = l_a_ind3 / l_s2;

                    l_a_ind2 -= l_ind2 * l_s1;
                    l_a_ind3 -= l_ind3 * l_s2;

                    int l_idx_input = l_ind3 + l_ind2 * l_d_i + l_k2 * l_d_i * l_d_i;

                    if (l_a_ind2 == 0 && l_a_ind3 == 0)
                    {
                        if (l_ind2 >= 0 && l_ind3 >= 0 && l_ind2 < l_d_i && l_ind3 < l_d_i)
                        {
                            float l_t1 = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                            float l_t2 = l_filter1[l_j3 - 4];

                            l_temp += l_t1 * l_t2;
                        }
                    }
                }
            }
        }
        if (l_in_rank_bias == 2 && l_in_shape_bias[1] > 1)
        {
            l_in_byte_offset_bias += 4 * l_k1;
        }

        float l_out = l_temp + asfloat(l_tensors.Load(l_in_byte_offset_bias));
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
    }
}
