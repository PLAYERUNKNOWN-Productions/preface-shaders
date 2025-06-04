// Copyright:   PlayerUnknown Productions BV

// Attributes specifically for this operation
struct conv_d_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_d_attribs_s p_attribs)
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

#define l_d_o l_d_i
#define l_d_f 3
#define l_s1 1
#define l_s2 1
#define l_d1 1
#define l_d2 1
#define l_p1 1
#define l_p2 1
#define l_df2 9

#define l_k2 (p_gid.z)
#define l_i1 (p_gtid.y)
#define l_i2 (p_gtid.x)

#define GROUP_SIZE 32

groupshared float l_filtersh[2048];

void conv_d_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1 + 20;
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2 + 12;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 20;

    uint l_in_ch = l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 * (1 + 1));
    uint l_d_i = l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 * (1 + 2));

    uint l_id = p_gtid.y * GROUP_SIZE + p_gtid.x;
    uint l_c = l_in_ch * l_d_f * l_d_f;

    for(;l_id < l_c; l_id += 1024)
    {
        l_filtersh[l_id] = asfloat(l_tensors.Load(l_in_byte_offset_filter + 4 * (l_k2 * l_c + l_id)));
    }
    GroupMemoryBarrierWithGroupSync();

    if (l_d_i == 32)
    {
        float l_temp = 0.0f;

        uint l_ii1 = l_i1 - l_p1;
        uint l_ii2 = l_i2 - l_p2;
        uint l_idx_input1 = l_ii2;
        uint l_idx_filter1 = 0;

        for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
        {
            for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
            {
                int l_ind1 = l_ii1 + l_j1;
                uint l_cond = ((l_ind1 >= 0)*(l_ind1 < l_d_i));
                int l_idx_input = l_idx_input1 + l_ind1 * l_d_i;
                float4 l_input = asfloat(l_tensors.Load4(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                uint l_idx_filter = l_idx_filter1 + l_j1 * l_d_f;

                for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
                {
                    int l_ind2 = l_ii2 + l_j2;
                    uint l_aa = (l_ind2 >= 0)*(l_ind2 < l_d_i) * l_cond;

                    l_temp += l_aa * l_input[l_j2] * l_filtersh[l_idx_filter+l_j2];
                }
            }
            l_idx_filter1 += l_df2;
            l_idx_input1 += 1024;
        }

        uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;
        float l_out = l_temp + asfloat(l_tensors.Load(l_in_byte_offset_bias + 4 * l_k2));
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
    }
    else if (l_d_i == 64)
    {
        float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias + 4 * l_k2));
        for (uint l_ii = 0; l_ii < 2; l_ii++)
        {
            for (uint l_jj = 0; l_jj < 2; l_jj++)
            {
                uint l_ii1 = l_i1 - l_p1 + 32 * l_ii;
                uint l_ii2 = l_i2 - l_p2 + 32 * l_jj;

                float l_temp = 0;

                uint l_idx_input1 = l_ii2;
                uint l_idx_filter1 = 0;

                for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
                {
                    for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
                    {
                        int l_ind1 = l_ii1 + l_j1;
                        uint l_cond = ((l_ind1 >= 0) * (l_ind1 < l_d_i));
                        int l_idx_input = l_idx_input1 + l_ind1 * l_d_i;
                        float4 l_input = asfloat(l_tensors.Load4(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                        uint l_idx_filter = l_idx_filter1 + l_j1 * l_d_f;

                        for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
                        {
                            int l_ind2 = l_ii2 + l_j2;
                            uint l_aa = (l_ind2 >= 0) * (l_ind2 < l_d_i) * l_cond;

                            l_temp += l_aa * l_input[l_j2] * l_filtersh[l_idx_filter + l_j2];
                        }
                    }
                    l_idx_filter1 += l_df2;
                    l_idx_input1 += 4096;
                }

                float l_out = l_temp + l_bias;

                uint l_idx_output = (l_i2 + 32 * l_jj) + (l_i1 + 32 * l_ii) * l_d_o + l_k2 * l_d_o * l_d_o;
                l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
            }
        }
    }
}
