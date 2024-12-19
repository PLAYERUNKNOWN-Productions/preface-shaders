// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

struct max_pool_attribs_s
{
    int4 m_size;
    int4 m_padding;
    int4 m_stride;
    int4 m_dilation;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out max_pool_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count != 20)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif // CML_KERNEL_ERROR_HANDLING

    p_attribs.m_size = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 32;
    p_attribs.m_padding = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_stride = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_dilation = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;

    return (l_byte_offset - p_byte_offset);
}

#define l_out_ch (l_out_shape[1])
#define l_in_ch (l_in_shape_input[1])
#define l_d_f (l_attribs.m_size.z)
#define l_s1 (l_attribs.m_stride.x)
#define l_s2 (l_attribs.m_stride.y)
#define l_d1 (l_attribs.m_dilation.x)
#define l_d2 (l_attribs.m_dilation.y)
#define l_p1 (l_attribs.m_padding.x)
#define l_p2 (l_attribs.m_padding.z)

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    max_pool_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    // Every Tensor starts with its shape
    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint l_k2 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;

    int l_d_o = (int)(l_out_shape[3]);
    int l_d_i = (int)(l_in_shape_input[3]);

    if (l_k2 < l_out_ch && l_i2 < l_out_shape[3] && l_i1 < l_out_shape[2])
    {
        int l_dim_0 = l_d_i;
        int l_dim_1 = l_dim_0;
        
        uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;

        float l_max_value = -FLT_MAX;

        for (uint l_j1 = 0; l_j1 < l_d_f; l_j1++)
        {
            for (uint l_j2 = 0; l_j2 < l_d_f; l_j2++)
            {
                int l_ind2 = l_i1 * l_s1 + l_j1 * l_d1 - l_p1;
                int l_ind3 = l_i2 * l_s2 + l_j2 * l_d2 - l_p2;

                int l_idx_input = l_ind3 + l_ind2 * l_dim_1;

                if (l_ind2 >= 0 && l_ind3 >= 0 && l_ind2 < l_d_i && l_ind3 < l_d_i)
                {
                    float l_t1 = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * (uint)l_idx_input));
                    
                    if (l_t1 > l_max_value)
                    {
                        l_max_value = l_t1;
                    }
                }
            }
        }
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_max_value));
    }
}
