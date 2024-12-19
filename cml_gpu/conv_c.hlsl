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
    p_attribs.m_stride = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

#define GROUP_SIZE 16
#define FLOAT_SIZE 4

groupshared float l_filter[2304];
groupshared float l_input[512];

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    uint l_in_ch = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + FLOAT_SIZE * 2));
    uint l_out_ch = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_3 + FLOAT_SIZE * 2));

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1 + 20;

    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2;
    uint2 l_in_shape_bias = asuint(l_tensors.Load2(l_in_byte_offset_bias + FLOAT_SIZE));
    l_in_byte_offset_bias += FLOAT_SIZE * (1 + 2);

    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 20;

    float l_value = 0.0f;
    uint l_tid = p_gtid.x * GROUP_SIZE + p_gtid.y;


    for (uint l_i = l_tid; l_i < l_in_ch * l_out_ch; l_i += GROUP_SIZE * GROUP_SIZE)
    {
        l_filter[l_i] = asfloat(l_tensors.Load(l_in_byte_offset_filter + FLOAT_SIZE * l_i));
    }
    if (l_tid < l_in_ch)
    {
        l_input[l_tid] = asfloat(l_tensors.Load(l_in_byte_offset_input + FLOAT_SIZE * l_tid));
    }
    else
        return;
    GroupMemoryBarrierWithGroupSync();


    for (uint l_k1 = 0; l_k1 < l_in_ch; l_k1++)
    {
        l_value += l_input[l_k1] * l_filter[l_tid * l_in_ch + l_k1];
    }

    // Get bias value
    // Do not add offset if shape_bias = [1, 1]
    if (l_in_shape_bias[1] > 1)
    {
        l_in_byte_offset_bias += FLOAT_SIZE * l_tid;
    }
    float l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));

    float l_out = l_value + l_bias;
    uint l_idx_output = l_tid;
    l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * l_idx_output, asuint(l_out));
}
