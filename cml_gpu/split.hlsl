// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    //
// uint m_tensor_offset_0; // Tensors
// uint m_tensor_offset_1;
// uint m_tensor_offset_2;
// uint m_tensor_offset_3;
// uint m_tensor_offset_4;
// uint m_tensor_offset_5;
// uint m_tensor_offset_6;
// uint m_tensor_offset_7;
// uint m_tensor_offset_8;
// uint m_tensor_offset_9;
// uint m_tensor_offset_10;
// uint m_tensor_offset_11;
// uint m_tensor_offset_12;
// uint m_tensor_offset_13;
// uint m_tensor_offset_14;
// uint m_tensor_offset_15;
// uint m_tensor_offset_16; // Tensors
// uint m_tensor_offset_17;
// uint m_tensor_offset_18;
// uint m_tensor_offset_19;
// uint m_tensor_offset_20;
// uint m_tensor_offset_21;
// uint m_tensor_offset_22;
// uint m_tensor_offset_23;
// uint m_tensor_offset_24;
// uint m_tensor_offset_25;
// uint m_tensor_offset_26;
// uint m_tensor_offset_27;
// uint m_tensor_offset_28;
// uint m_tensor_offset_29;
// uint m_tensor_offset_30;
// uint m_tensor_offset_31;

// UP TO 15 TENSORS CAN BE CONCATENATED
#define MAX_TENSORS 32
#define GROUP_SIZE 16

#define GROUP_SIZE 16

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_shape_tensor[MAX_TENSORS][4];
    uint l_byte_offset_tensor[MAX_TENSORS] =
    {
        l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1,
        l_meta_data.m_tensor_offset_2, l_meta_data.m_tensor_offset_3,
        l_meta_data.m_tensor_offset_4, l_meta_data.m_tensor_offset_5,
        l_meta_data.m_tensor_offset_6, l_meta_data.m_tensor_offset_7,
        l_meta_data.m_tensor_offset_8, l_meta_data.m_tensor_offset_9,
        l_meta_data.m_tensor_offset_10, l_meta_data.m_tensor_offset_11,
        l_meta_data.m_tensor_offset_12, l_meta_data.m_tensor_offset_13,
        l_meta_data.m_tensor_offset_14, l_meta_data.m_tensor_offset_15,
        l_meta_data.m_tensor_offset_16, l_meta_data.m_tensor_offset_17,
        l_meta_data.m_tensor_offset_18, l_meta_data.m_tensor_offset_19,
        l_meta_data.m_tensor_offset_20, l_meta_data.m_tensor_offset_21,
        l_meta_data.m_tensor_offset_22, l_meta_data.m_tensor_offset_23,
        l_meta_data.m_tensor_offset_24, l_meta_data.m_tensor_offset_25,
        l_meta_data.m_tensor_offset_26, l_meta_data.m_tensor_offset_27,
        l_meta_data.m_tensor_offset_28, l_meta_data.m_tensor_offset_29,
        l_meta_data.m_tensor_offset_30, l_meta_data.m_tensor_offset_31
    };
    uint l_ratio[MAX_TENSORS - 1];

    // Save byte offsets, shapes, and flattened sizes
    for (uint l_i = 0; l_i < l_meta_data.m_tensor_count; l_i++)
    {
        uint l_rank = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for (uint l_j = 0; l_j < l_rank; l_j++)
        {
            l_shape_tensor[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + l_j)));
        }
        for (uint l_k = l_rank; l_k < 4; l_k++)
        {
            l_shape_tensor[l_i][l_k] = 1;
        }

        l_byte_offset_tensor[l_i] += 4 * (1 + l_rank);
    }

    uint l_tid_x = p_gid.x * GROUP_SIZE + p_gtid.x;
    uint l_tid_y = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_tid_z = p_gid.z;

    uint l_idx[4] = { 0, l_tid_z, l_tid_y, l_tid_x };
    uint l_axis[4] = { 0, 0, 0, 0 };
    uint l_axes_index = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4));
    l_axis[l_axes_index] = 1;
    uint l_bin = 0;

    for (uint l_i = 0; l_i < l_meta_data.m_attrib_count - 1; l_i++)
    {
        l_ratio[l_i] = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (2 + l_i)));
    }

    uint l_boundary = 0;
    uint l_idx_adj;

    for (uint l_i = 0; l_i < l_meta_data.m_tensor_count - 1; l_i++)
    {
        l_boundary += l_ratio[l_i];

        if (l_idx[l_axes_index] < l_boundary)
        {
            l_bin = l_i + 1;
            l_idx_adj = l_idx[l_axes_index] - (l_boundary - l_ratio[l_i]);
            break;
        }
    }

    // identified bin
    // compute adjusted index and store

    uint l_temp1, l_temp2, l_temp3;
    l_temp1 = (1 - l_axis[1]) * l_idx[1] + l_axis[1] * l_idx_adj;
    l_temp2 = (1 - l_axis[2]) * l_idx[2] + l_axis[2] * l_idx_adj;
    l_temp3 = (1 - l_axis[3]) * l_idx[3] + l_axis[3] * l_idx_adj;

    uint l_idx_out = l_temp1 * l_shape_tensor[l_bin][3] * l_shape_tensor[l_bin][2]
                   + l_temp2 * l_shape_tensor[l_bin][3]
                   + l_temp3;
    uint l_idx_in = l_idx[1] * l_shape_tensor[0][3] * l_shape_tensor[0][2]
                  + l_idx[2] * l_shape_tensor[0][3]
                  + l_idx[3];

    if (l_tid_z < l_shape_tensor[0][1] && l_tid_y < l_shape_tensor[0][2] && l_tid_x < l_shape_tensor[0][3])
    {
        float l_out = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx_in));

        l_tensors.Store(l_byte_offset_tensor[l_bin] + 4 * l_idx_out, asuint(l_out));
    }
}
