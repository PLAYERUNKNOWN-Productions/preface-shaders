// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_rank_a = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_0)));
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    float l_attr_factor_x = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 4));
    float l_attr_factor_y = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 8));

    uint l_k1 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;
    uint l_n_output = l_out_shape[0] * l_out_shape[1] * l_out_shape[2] * l_out_shape[3];
       
    uint l_idx_out = l_k1 * l_out_shape[2] * l_out_shape[3] + l_i1 * l_out_shape[3] + l_i2;

    if (l_idx_out < l_n_output && l_i1 < l_out_shape[2] && l_i2 < l_out_shape[3])
    {
        uint l_idx_y = l_k1;
        uint l_idx_z = l_i1;
        uint l_idx_w = l_i2;

        uint l_ind_in = l_idx_y * l_in_shape_a[2] * l_in_shape_a[3]
                      + (uint)(l_idx_z / l_attr_factor_x) * l_in_shape_a[3]
                      + (uint)(l_idx_w / l_attr_factor_y);

        float l_out = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_ind_in));
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_out, asuint(l_out));
    }
}
