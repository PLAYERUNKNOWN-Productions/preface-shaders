// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 32
#define l_d_o (l_out_shape[3])

struct leaky_relu_attribs_s
{
    float m_alpha;
};

float leaky_relu(in float p_a, in float p_alpha)
{
    return ((p_a > 0) ? p_a : p_alpha * p_a);
}

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out leaky_relu_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count != 1)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif // CML_KERNEL_ERROR_HANDLING

    p_attribs.m_alpha = asfloat(p_attrib_buffer.Load(l_byte_offset));
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    leaky_relu_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint l_k2 = p_gid.z;
    uint l_i10 = p_gtid.y;
    uint l_i20 = p_gtid.x;

    for (uint l_i1 = l_i10; l_i1 < l_out_shape[2]; l_i1 += GROUP_SIZE)
    {
        for (uint l_i2 = l_i20; l_i2 < l_out_shape[3]; l_i2 += GROUP_SIZE)
        {
            uint l_idx_output = l_i2 + l_i1 * l_d_o + l_k2 * l_d_o * l_d_o;
            float l_out = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_idx_output));
            if (l_out < 0.0f)
            {
                l_out *= l_attribs.m_alpha;
            }
            l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
        }
    }
}
