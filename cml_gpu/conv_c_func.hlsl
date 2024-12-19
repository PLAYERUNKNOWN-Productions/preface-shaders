// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

// Attributes specifically for this operation
struct conv_c_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_c_attribs_s p_attribs)
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

#define l_d_o 32
#define l_d_i 32
#define l_d_f 1
#define l_s1 1
#define l_s2 1
#define l_d1 1
#define l_d2 1
#define l_p1 0
#define l_p2 0

#define l_k2 (p_gid.z)
#define l_i1 (p_gtid.y)
#define l_i2 (p_gtid.x)

#define GROUP_SIZE 32
#define FLOAT_SIZE 4
#define l_d_i_sqr (l_d_i * l_d_i)

void conv_c_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    uint l_in_ch = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 4 * 2));
    uint l_out_ch = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_3 + 4 * 2));

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1 + 20;
    
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2;
    uint2 l_in_shape_bias = asuint(l_tensors.Load2(l_in_byte_offset_bias + FLOAT_SIZE));
    l_in_byte_offset_bias += FLOAT_SIZE * (1 + 2);

    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 20;

    float l_value = 0.0f;
    uint l_idx_filter = l_k2 * l_in_ch;
    uint l_idx_input = l_i2 + l_i1 * l_d_i;

    for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
    {
        float l_filter = asfloat(l_tensors.Load(l_in_byte_offset_filter + FLOAT_SIZE * l_idx_filter));
        float l_input = asfloat(l_tensors.Load(l_in_byte_offset_input + FLOAT_SIZE * l_idx_input));

        l_idx_filter++;
        l_value += l_input * l_filter;
        l_idx_input += l_d_i_sqr;
    }

    // Get bias value
    // Do not add offset if shape_bias = [1, 1]
    if (l_in_shape_bias[1] > 1)
    {
        l_in_byte_offset_bias += FLOAT_SIZE * l_k2;
    }
    float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));

    float l_out = l_value + l_bias;
    uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;
    l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
}
