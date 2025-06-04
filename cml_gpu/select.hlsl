// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 4
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // lower bound
// uint m_tensor_offset_2; // upper bound
// uint m_tensor_offset_3; // Output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
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
    uint4 l_in_shape_input;
    uint4 l_in_shape_low;
    uint4 l_in_shape_high;
    uint4 l_out_shape_output;

    uint l_in_rank_input = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_0)));
    uint l_in_rank_low = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_1)));
    uint l_in_rank_high = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_2)));
    uint l_out_rank_output = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_3)));

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    uint l_in_byte_offset_low = l_meta_data.m_tensor_offset_1;
    uint l_in_byte_offset_high = l_meta_data.m_tensor_offset_2;
    uint l_out_byte_offset_output = l_meta_data.m_tensor_offset_3;

    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);
    l_in_byte_offset_low += tensor_shape(l_tensors, l_in_byte_offset_low, l_in_shape_low);
    l_in_byte_offset_high += tensor_shape(l_tensors, l_in_byte_offset_high, l_in_shape_high);
    l_out_byte_offset_output += tensor_shape(l_tensors, l_out_byte_offset_output, l_out_shape_output);

    uint l_dim_x[4];
    uint l_dim_y[4];
    uint l_dim_z[4];
    uint l_broadcast_x[4];
    uint l_broadcast_y[4];

    // match dimensions of x and y with z's
    for (int l_i = 0; l_i < 4; l_i++)
    {
        // unsqueeze z to be rank 4
        if (l_out_rank_output > l_i)
        {
            l_dim_z[l_i] = l_out_shape_output[l_i];
        }
        else
        {
            l_dim_z[l_i] = 1;
        }

        // check if broadcasting is needed on any dimensions
        l_broadcast_x[l_i] = l_in_rank_low <= l_i || l_in_shape_low[l_i] < l_dim_z[l_i];
        l_broadcast_y[l_i] = l_in_rank_high <= l_i || l_in_shape_high[l_i] < l_dim_z[l_i];

        if (l_broadcast_x[l_i])
        {
            l_dim_x[l_i] = 1;
        }
        else
        {
            l_dim_x[l_i] = l_in_shape_low[l_i];
        }

        if (l_broadcast_y[l_i])
        {
            l_dim_y[l_i] = 1;
        }
        else
        {
            l_dim_y[l_i] = l_in_shape_high[l_i];
        }
    }

    uint l_id0 = (uint)(p_gid.z / l_out_shape_output[1]);
    uint l_id1 = p_gid.z % l_out_shape_output[1];
    uint l_id2 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_id0 < l_out_shape_output[0] && l_id1 < l_out_shape_output[1] && l_id2 < l_out_shape_output[2] && l_id3 < l_out_shape_output[3])
    {
        uint l_idx_z = l_id0 * (l_dim_z[1] * l_dim_z[2] * l_dim_z[3])
                     + l_id1 * (l_dim_z[2] * l_dim_z[3])
                     + l_id2 * l_dim_z[3]
                     + l_id3;

        uint l_idx_x = l_id0 * (1 - l_broadcast_x[0]) * (l_dim_x[1] * l_dim_x[2] * l_dim_x[3])
                     + l_id1 * (1 - l_broadcast_x[1]) * (l_dim_x[2] * l_dim_x[3])
                     + l_id2 * (1 - l_broadcast_x[2]) * l_dim_x[3]
                     + l_id3 * (1 - l_broadcast_x[3]);

        uint l_idx_y = l_id0 * (1 - l_broadcast_y[0]) * (l_dim_y[1] * l_dim_y[2] * l_dim_y[3])
                     + l_id1 * (1 - l_broadcast_y[1]) * (l_dim_y[2] * l_dim_y[3])
                     + l_id2 * (1 - l_broadcast_y[2]) * l_dim_y[3]
                     + l_id3 * (1 - l_broadcast_y[3]);

        float l_true_value = asfloat(l_tensors.Load(l_in_byte_offset_low + 4 * l_idx_x));
        float l_false_value = asfloat(l_tensors.Load(l_in_byte_offset_high + 4 * l_idx_y));
        int l_z = asint(l_tensors.Load(l_in_byte_offset_input + 4 * l_idx_z));

        float l_out = l_z ? l_true_value : l_false_value;

        // Store result
        l_tensors.Store(l_out_byte_offset_output + 4 * l_idx_z, asuint(l_out));
    }
}
