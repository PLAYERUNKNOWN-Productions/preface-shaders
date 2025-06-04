// Copyright:   PlayerUnknown Productions BV

/* Background info

All tensors store data non-interleaved.
So iteration is like this:

uint l_tensor_shape[4] = {1, 1, 32, 32};
float l_tensor[32 * 32 * 1 * 1];
for (size_t l_ch = 0; l_ch < l_tensor_shape[1]; ++l_ch)
{
    for (size_t l_row = 0; l_row < l_tensor_shape[2]; ++l_row)
    {
        for (size_t l_col = 0; l_col < l_tensor_shape[3]; ++l_col)
        {
            uint l_index = (l_ch * l_tensor_shape[3] * l_tensor_shape[2]) + (l_row * l_tensor_shape[3]) + l_col;
        }
    }
}
*/

/* Required meta-data
- Dispatch dimensions
- Tensor data-type (int or float)
- Tensor dimensions
- Offset in l_tensors for each tensor
- Attributes

How can we prevent mapping a buffer to specify offsets at dispatch time?
Using an indirect buffer?
At startup determine the offset for all operations in the agent model.
Store them in indirect buffer. But how is a value in that buffer matched with the dispatch?
Basically the same offset issue again.
Use DispatchIndirect()!
It takes a buffer and an offset for the buffer to specify where to start!
https://docs.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-dispatchindirect
But only provides for the X, Y, Z params of Dispatch(), no custom values  :-(
*/

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

DEF_THREAD_GROUP_SIZE_BINARY
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    if (l_meta_data.m_attrib_count != 0)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }

    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_in_shape_b;
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_b += tensor_shape(l_tensors, l_in_byte_offset_b, l_in_shape_b);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_2;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint4 l_nbroadcast_x = uint4(1 - (l_in_shape_a[0] < l_out_shape[0]),
                                 1 - (l_in_shape_a[1] < l_out_shape[1]),
                                 1 - (l_in_shape_a[2] < l_out_shape[2]),
                                 1 - (l_in_shape_a[3] < l_out_shape[3]));
    uint4 l_nbroadcast_y = uint4(1 - (l_in_shape_b[0] < l_out_shape[0]),
                                 1 - (l_in_shape_b[1] < l_out_shape[1]),
                                 1 - (l_in_shape_b[2] < l_out_shape[2]),
                                 1 - (l_in_shape_b[3] < l_out_shape[3]));

    uint l_id0 = p_gid.z / l_out_shape[1];
    uint l_id1 = p_gid.z % l_out_shape[1];
    uint l_id2 = p_gid.y * GROUP_SIZE_BINARY_Y + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE_BINARY_X + p_gtid.x;

    if (l_id2 < l_out_shape[2] && l_id3 < l_out_shape[3])
    {
        uint l_idx_z = l_id0 * l_out_shape[1] * l_out_shape[2] * l_out_shape[3]
                     + l_id1 * l_out_shape[2] * l_out_shape[3]
                     + l_id2 * l_out_shape[3]
                     + l_id3;

        uint l_idx_x = l_id0 * l_nbroadcast_x[0] * l_in_shape_a[1] * l_in_shape_a[2] * l_in_shape_a[3]
                     + l_id1 * l_nbroadcast_x[1] * l_in_shape_a[2] * l_in_shape_a[3]
                     + l_id2 * l_nbroadcast_x[2] * l_in_shape_a[3]
                     + l_id3 * l_nbroadcast_x[3];

        uint l_idx_y = l_id0 * l_nbroadcast_y[0] * l_in_shape_b[1] * l_in_shape_b[2] * l_in_shape_b[3]
                     + l_id1 * l_nbroadcast_y[1] * l_in_shape_b[2] * l_in_shape_b[3]
                     + l_id2 * l_nbroadcast_y[2] * l_in_shape_b[3]
                     + l_id3 * l_nbroadcast_y[3];

        float l_x = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * l_idx_x));
        float l_y = asfloat(l_tensors.Load(l_in_byte_offset_b + 4 * l_idx_y));
        float l_z = l_x + l_y;

        // Store result
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_z, asuint(l_z));
    }
}
