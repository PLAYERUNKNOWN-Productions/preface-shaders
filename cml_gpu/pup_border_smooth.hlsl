// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

#define SIZE_FLOAT 4

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Every Tensor starts with its shape
    uint4 l_in_shape = uint4(1, 1, 32, 32);
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0 + 5 * SIZE_FLOAT;

    uint4 l_out_shape = uint4(1, 1, 64, 64);
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1 + 5 * SIZE_FLOAT;

    uint l_out_byte_offset = l_meta_data.m_tensor_offset_2 + 5 * SIZE_FLOAT;

    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;

    uint l_idx_out = l_i1 * l_out_shape[3] + l_i2;

    float l_x = (float)l_i1 / 2;

    int l_x1 = floor(l_x) - (l_i1 == l_out_shape[2] - 2);
    int l_x2 = l_x1 + 1;

    float l_y = (float)l_i2 / 2;

    int l_y1 = floor(l_y) - (l_i2 == l_out_shape[3] - 2);
    int l_y2 = l_y1 + 1;

    int l_idx_output = SIZE_FLOAT * (l_i2 + l_out_shape[3] * l_i1);

    // copy upresed to output
    float l_im64 = asfloat(l_tensors.Load(l_in_byte_offset_b + l_idx_output));

    // postprocessing starts here
    int l_idx_input11 = SIZE_FLOAT * (l_y1 + l_in_shape[3] * l_x1);
    int l_idx_input12 = SIZE_FLOAT * (l_y2 + l_in_shape[3] * l_x1);
    int l_idx_input21 = SIZE_FLOAT * (l_y1 + l_in_shape[3] * l_x2);
    int l_idx_input22 = SIZE_FLOAT * (l_y2 + l_in_shape[3] * l_x2);

    float l_f11 = asfloat(l_tensors.Load(l_in_byte_offset_a + l_idx_input11));
    float l_f12 = asfloat(l_tensors.Load(l_in_byte_offset_a + l_idx_input12));
    float l_f21 = asfloat(l_tensors.Load(l_in_byte_offset_a + l_idx_input21));
    float l_f22 = asfloat(l_tensors.Load(l_in_byte_offset_a + l_idx_input22));

    float l_out = l_f11 * (l_x2 - l_x) * (l_y2 - l_y)
                + l_f21 * (l_x - l_x1) * (l_y2 - l_y)
                + l_f12 * (l_x2 - l_x) * (l_y - l_y1)
                + l_f22 * (l_x - l_x1) * (l_y - l_y1);

    int l_border = l_i1 <= 5 || l_i1 >= 57 || l_i2 <= 5 || l_i2 >= 57;
    l_out = l_border * l_out + (1 - l_border) * l_im64;

    l_tensors.Store(l_out_byte_offset + l_idx_output, asuint(l_out));
}
