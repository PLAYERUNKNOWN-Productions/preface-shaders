// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 5
// uint m_tensor_offset_0;
// uint m_tensor_offset_1;
// uint m_tensor_offset_2;
// uint m_tensor_offset_3;
// uint m_tensor_offset_4;

// UP TO 15 TENSORS CAN BE ADDED

#define MAX_TENSORS 16
DEF_THREAD_GROUP_SIZE_BINARY
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_byte_offset_tensor[MAX_TENSORS] = {
        l_meta_data.m_tensor_offset_0,  l_meta_data.m_tensor_offset_1,  l_meta_data.m_tensor_offset_2,  l_meta_data.m_tensor_offset_3,
        l_meta_data.m_tensor_offset_4,  l_meta_data.m_tensor_offset_5,  l_meta_data.m_tensor_offset_6,  l_meta_data.m_tensor_offset_7,
        l_meta_data.m_tensor_offset_8,  l_meta_data.m_tensor_offset_9,  l_meta_data.m_tensor_offset_10, l_meta_data.m_tensor_offset_11,
        l_meta_data.m_tensor_offset_12, l_meta_data.m_tensor_offset_13, l_meta_data.m_tensor_offset_14, l_meta_data.m_tensor_offset_15};

    // Output tensor
    uint l_n_out = l_meta_data.m_tensor_count - 1;
    uint4 l_out_shape;
    l_byte_offset_tensor[l_n_out] += tensor_shape(l_tensors, l_byte_offset_tensor[l_n_out], l_out_shape);

    uint l_id0 = p_gid.z / l_out_shape[1];
    uint l_id1 = p_gid.z % l_out_shape[1];
    uint l_id2 = p_gid.y * GROUP_SIZE_BINARY_Y + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE_BINARY_X + p_gtid.x;

    if (l_id2 < l_out_shape[2] && l_id3 < l_out_shape[3])
    {
        // Save byte offsets, shapes, and flattened sizes
        float l_sum = 0;

        for (uint l_i = 0; l_i < l_n_out; l_i++)
        {
            uint4 l_in_shape_a;
            l_byte_offset_tensor[l_i] += tensor_shape(l_tensors, l_byte_offset_tensor[l_i], l_in_shape_a);

            uint4 l_nbroadcast_x = uint4(1 - (l_in_shape_a[0] < l_out_shape[0]),
                                         1 - (l_in_shape_a[1] < l_out_shape[1]),
                                         1 - (l_in_shape_a[2] < l_out_shape[2]),
                                         1 - (l_in_shape_a[3] < l_out_shape[3]));

            uint l_idx_x = l_id0 * l_nbroadcast_x[0] * l_in_shape_a[1] * l_in_shape_a[2] * l_in_shape_a[3]
                         + l_id1 * l_nbroadcast_x[1] * l_in_shape_a[2] * l_in_shape_a[3]
                         + l_id2 * l_nbroadcast_x[2] * l_in_shape_a[3]
                         + l_id3 * l_nbroadcast_x[3];

            float l_x = asfloat(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * l_idx_x));
            l_sum += l_x;
        }

        uint l_idx_z = l_id0 * l_out_shape[1] * l_out_shape[2] * l_out_shape[3]
                     + l_id1 * l_out_shape[2] * l_out_shape[3]
                     + l_id2 * l_out_shape[3]
                     + l_id3;

        // Store result
        l_tensors.Store(l_byte_offset_tensor[l_n_out] + 4 * l_idx_z, asuint(l_sum));
    }
}
