// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Output

#define SIZE_FLOAT 4   // sizeof(float)

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Every Tensor starts with its shape
    uint4 l_in_shape;
    uint l_in_byte_offset = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset += tensor_shape(l_tensors, l_in_byte_offset, l_in_shape);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);
    
    uint l_k1 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;
    int l_n_output = l_out_shape[1] * l_out_shape[2] * l_out_shape[3];       
    int l_idx_out = l_k1 * l_out_shape[2] * l_out_shape[3] + l_i1 * l_out_shape[3] + l_i2;
    
    // align_corner = true
    float l_y = float(l_i1 * (l_in_shape[2] - 1)) / float(l_out_shape[2] - 1);
    float l_x = float(l_i2 * (l_in_shape[3] - 1)) / float(l_out_shape[3] - 1);
    int l_y1 = floor(l_y);
    int l_x1 = floor(l_x);
    int l_y2 = l_y1 + 1;
    int l_x2 = l_x1 + 1;
    
    if (l_x1 == l_in_shape[3] - 1)
    {
        l_x1--;
        l_x2--;
    }
    
    if (l_y1 == l_in_shape[2] - 1)
    {
        l_y1--;
        l_y2--;
    }
    
    uint l_idx_input11 = SIZE_FLOAT * (l_x1 + l_in_shape[3] * (l_y1 + l_in_shape[2] * l_k1));
    uint l_idx_input12 = SIZE_FLOAT * (l_x2 + l_in_shape[3] * (l_y1 + l_in_shape[2] * l_k1));
    uint l_idx_input21 = SIZE_FLOAT * (l_x1 + l_in_shape[3] * (l_y2 + l_in_shape[2] * l_k1));
    uint l_idx_input22 = SIZE_FLOAT * (l_x2 + l_in_shape[3] * (l_y2 + l_in_shape[2] * l_k1));

    float l_f11 = asfloat(l_tensors.Load(l_in_byte_offset + l_idx_input11));
    float l_f12 = asfloat(l_tensors.Load(l_in_byte_offset + l_idx_input12));
    float l_f21 = asfloat(l_tensors.Load(l_in_byte_offset + l_idx_input21));
    float l_f22 = asfloat(l_tensors.Load(l_in_byte_offset + l_idx_input22));

    float l_out = l_f11 * (l_x2 - l_x) * (l_y2 - l_y)
                + l_f12 * (l_x - l_x1) * (l_y2 - l_y)
                + l_f21 * (l_x2 - l_x) * (l_y - l_y1)
                + l_f22 * (l_x - l_x1) * (l_y - l_y1);

    if (l_k1 < l_out_shape[1] && l_i1 < l_out_shape[2] && l_i2 < l_out_shape[3])
    {
        l_tensors.Store(l_out_byte_offset + SIZE_FLOAT * l_idx_out, asuint(float(l_out)));
    }
}
