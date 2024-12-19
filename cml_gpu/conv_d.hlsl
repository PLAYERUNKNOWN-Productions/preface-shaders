// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 4
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Filter
// uint m_tensor_offset_2; // Bias
// uint m_tensor_offset_3; // Output

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
    p_attribs.m_stride = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

#define GROUP_SIZE 8
#define FLOAT_SIZE 4
#define UINT_SIZE 4
#define GROUP_SIZE_X 8

#define l_in_ch (l_in_shape_filter[1])
#define l_out_ch (l_in_shape_filter[0])
#define l_W (l_in_shape_input[3])

groupshared float l_filter[2304];
// 1x1 filter assumed

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
[numthreads(GROUP_SIZE_X, GROUP_SIZE, GROUP_SIZE)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
    uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    uint4 l_in_shape_input = asuint(l_tensors.Load4(l_meta_data.m_tensor_offset_0 + UINT_SIZE));
    uint4 l_in_shape_filter = asuint(l_tensors.Load4(l_meta_data.m_tensor_offset_1 + UINT_SIZE));
    uint2 l_in_shape_bias = asuint(l_tensors.Load2(l_meta_data.m_tensor_offset_2 + UINT_SIZE)); // rank2 assumed

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1 + 20;
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2 + 12;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 20;
        
    uint l_x = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_y = p_gid.z * GROUP_SIZE + p_gtid.z;
    uint l_k2 = p_gid.x * GROUP_SIZE_X + p_gtid.x;

    uint l_tid = p_gtid.y * GROUP_SIZE + p_gtid.z;
    uint l_image_size = l_W * l_W;


    if (l_k2 >= l_in_shape_filter[0])
        return;

    // copy filter to groupshared
    for (uint l_i = l_tid; l_i < l_in_ch; l_i += GROUP_SIZE * GROUP_SIZE)
    {
        l_filter[l_k2 * l_in_ch + l_i] = asfloat(l_tensors.Load(l_in_byte_offset_filter + FLOAT_SIZE * (l_k2 * l_in_ch + l_i)));
    }

    GroupMemoryBarrierWithGroupSync();

    if (l_x >= l_in_shape_input[2] || l_y >= l_in_shape_input[3])
    {
        return;
    }

    uint l_idx_output = l_x * l_W + l_y;
    uint l_offset = l_k2 * l_in_ch;
    float l_value = 0.0f;

    for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
    {
        // READ INPUT
        float l_input = asfloat(l_tensors.Load(l_in_byte_offset_input + FLOAT_SIZE * (l_k1 * l_image_size + l_idx_output)));

        // ADD FOR FOUR FILTER
        l_value += l_input * l_filter[l_offset + l_k1];
    }

    // Get bias value
    // Do not add offset if shape_bias = [1, 1]

    if (l_in_shape_bias[1] > 1)
    {
        l_value += asfloat(l_tensors.Load(l_in_byte_offset_bias + FLOAT_SIZE * l_k2));
    }
    else
    {
        l_value += asfloat(l_tensors.Load(l_in_byte_offset_bias));
    }

    l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * (l_k2 * l_image_size + l_idx_output), asuint(l_value));
}
