// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

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
#define MAX_GROUPSHARED_FLOAT 8192

//-----------------------------------------------------------------------------
// ASSUMPTIONS
//-----------------------------------------------------------------------------
//
// l_d_f <= 4
// l_d1 == l_d2 == 1
// l_attribs.m_padding[0] = l_attribs.m_padding[1]
// l_attribs.m_padding[2] = l_attribs.m_padding[3]
//

groupshared float l_filter[MAX_GROUPSHARED_FLOAT];

void conv_func(uint3 p_gid, uint3 p_dtid, uint3 p_gtid, uint p_gi)
{
    CML_GET_BUFFERS;

    //   We cannot enable below when we have GroupMemoryBarrierWithGroupSync();
    //CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    conv_attribs_s l_attribs;
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

    uint l_i1 = p_gid.y * GROUP_SIZE_AT_Y + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE_AT_X + p_gtid.x;
    uint l_k2 = p_gid.z;
    int l_ind1_0 = l_i1 * l_s1 - l_p1;
    int l_ind2_0 = l_i2 * l_s2 - l_p2;
    uint l_idx_input0 = (uint)(l_k2 * (uint)l_attribs.m_groups / l_out_ch) * l_in_shape_filter[1] * l_d_i * l_d_i;
    uint l_idx_filter0 = 0;
    uint l_d_i_sqr = l_d_i * l_d_i;
    uint l_d_f_sqr = l_d_f * l_d_f;
    uint l_filter_volume = l_in_shape_filter[1] * l_d_f_sqr;
    uint l_filter_offset = l_k2 * l_filter_volume;
    uint l_inc = 4 * GROUP_SIZE_AT_X * GROUP_SIZE_AT_Y;

    // Copy entire filter [l_k2,:,:,:] to groupshared
    for(uint l_i = 4 * (p_gtid.y * GROUP_SIZE_AT_X + p_gtid.x); l_i < l_filter_volume; l_i += l_inc)
    {
        float4 l_buffer = asfloat(l_tensors.Load4(l_in_byte_offset_filter + FLOAT_SIZE * (l_i + l_filter_offset)));

        l_filter[l_i] = l_buffer.x;
        l_filter[l_i+1] = l_buffer.y;
        l_filter[l_i+2] = l_buffer.z;
        l_filter[l_i+3] = l_buffer.w;
    }  
    GroupMemoryBarrierWithGroupSync();

    if (l_i1 < l_out_shape[2] && l_i2 < l_out_shape[3])
    {
        float l_out = 0.0f;

        // input channel
        for (uint l_k1 = 0; l_k1 < l_in_shape_filter[1]; l_k1++)
        {
            // 2D filter 
            for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
            {
                int l_ind1 = l_ind1_0 + l_j1;
                int l_idx_input = l_idx_input0 + l_ind1 * l_d_i;
                uint l_idx_filter = l_idx_filter0 + l_j1 * l_d_f;
                uint l_ind1_within = l_ind1 >= 0 && l_ind1 < l_d_i;
                float4 l_input_buffer = asfloat(l_tensors.Load4(l_in_byte_offset_input + FLOAT_SIZE * (uint)(l_idx_input + l_ind2_0)));

                // 2D filter
                for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
                {
                    int l_ind2 = l_ind2_0 + l_j2;
                
                    // If index is not outside input dimension
                    if (l_ind1_within && l_ind2 >= 0 && l_ind2 < l_d_i)
                    {
                        l_out += l_input_buffer[l_j2] * l_filter[l_idx_filter + l_j2];
                    }
                }
            }

            // Increment index for the next input channel
            l_idx_input0 += l_d_i_sqr;
            l_idx_filter0 += l_d_f_sqr;
        }

        // Get bias value
        // Do not add offset if shape_bias = [1, 1]
        if (l_in_shape_bias[1] > 1)
        {
            l_in_byte_offset_bias += FLOAT_SIZE * l_k2;
        }
        float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));

        uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o; 
        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * l_idx_output, asuint(l_out + l_bias));
    }
}
