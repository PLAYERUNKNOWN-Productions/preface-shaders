// Copyright:   PlayerUnknown Productions BV

// Attributes specifically for this operation
struct conv_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_attribs_s p_attribs)
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
#endif

    p_attribs.m_padding = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_stride = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

#define GROUP_SIZE 16

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

#define l_k2 (p_gid.z)
#define l_i1 (p_gid.y * GROUP_SIZE + p_gtid.y)
#define l_i2 (p_gid.x * GROUP_SIZE + p_gtid.x)

groupshared float l_filter[4096];
void conv_f_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    // We cannot enable below when we have GroupMemoryBarrierWithGroupSync();
    //PP_CHECK_KERNEL_ERROR;

    // Get the attributes
    conv_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint4 l_in_shape_filter;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_filter += tensor_shape(l_tensors, l_in_byte_offset_filter, l_in_shape_filter);

    uint4 l_in_shape_bias;
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2;
    uint l_in_rank_bias = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2));
    l_in_byte_offset_bias += 4 * (1 + l_in_rank_bias);

    for (uint l_i = 0; l_i < l_in_rank_bias; l_i++)
    {
        l_in_shape_bias[l_i] = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_2 + 4 * (1 + l_i)));
    }

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    float l_temp = 0.0f;

    uint l_tensor_size = l_in_shape_input[1] * (l_d_i + 2) * (l_d_i + 2);

    l_tensor_size = l_in_shape_filter[1] * l_d_f * l_d_f;
    uint l_c = l_k2 * l_d_f * l_d_f * l_in_ch;

    for (uint l_i = 4 * (p_gtid.y * 16 + p_gtid.x); l_i < l_tensor_size; l_i += 1024)
    {
        float4 l_temp1 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_i + l_c)));
        l_filter[l_i] = l_temp1.x;
        l_filter[l_i + 1] = l_temp1.y;
        l_filter[l_i + 2] = l_temp1.z;
        l_filter[l_i + 3] = l_temp1.w;
    }
    GroupMemoryBarrierWithGroupSync();

    int l_ind2_0 = l_i1 * l_s1;
    int l_ind3_0 = l_i2 * l_s2;
    uint l_idx_input0 = 0;
    uint l_idx_filter0 = 0;
    uint l_di2 = (l_d_i + 2) * (l_d_i + 2);
    uint l_df2 = l_d_f * l_d_f;

    for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
    {
        for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
        {
            int l_ind2 = l_ind2_0 + l_j1;
            int l_idx_input = l_idx_input0 + l_ind2 * l_d_i;
            uint l_idx_filter = l_idx_filter0 + l_j1 * l_d_f;

            for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
            {
                int l_ind3 = l_ind3_0 + l_j2;

                // Change this scratch
                float l_t1 = asfloat(l_scratch.Load(4 * (uint)(l_idx_input + l_ind3)));
                float l_t2 = l_filter[l_idx_filter + l_j2];
                l_temp += l_t1 * l_t2;
            }
        }
        l_idx_input0 += l_di2;
        l_idx_filter0 += l_df2;
    }

    uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;

    if (l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        if ((l_in_rank_bias > 1) && (l_in_shape_bias[1] != 1))
        {
            l_in_byte_offset_bias += 4 * l_k2;
        }

        float l_out = l_temp + asfloat(l_tensors.Load(l_in_byte_offset_bias));
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
    }

}
