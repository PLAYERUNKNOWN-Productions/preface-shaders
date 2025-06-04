// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

groupshared uint l_output[4096];

#define GROUP_SIZE 1024

[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    uint l_ov = 2; // rank - 4
    uint l_offset = 12; //4 * (1 + l_ov);

    uint4 l_dim_input = l_tensors.Load4(l_meta_data.m_tensor_offset_0 + l_offset);
    uint4 l_dim_output = l_tensors.Load4(l_meta_data.m_tensor_offset_1 + l_offset);
    uint4 l_axes = l_attributes.Load4(l_meta_data.m_attrib_offset + l_offset);

    uint l_n_output = shape_size(l_dim_input);  // 4096?

    l_axes -= l_ov;
    l_offset += 16;  // dimensions already read

    uint l_ii = p_gid.x;

    uint l_offset2 = l_offset + 4 * l_ii * l_n_output;

    // treat as rank 4 tensor
    uint l_tensor_in_offset = l_meta_data.m_tensor_offset_0 + l_offset2;
    uint l_tensor_out_offset = l_meta_data.m_tensor_offset_1 + l_offset2;

    // each thread takes care of 4 elements
    uint l_ind = 4 * p_gtid.x;
    uint4 l_input4 = l_tensors.Load4(l_tensor_in_offset + 4 * l_ind);

    for (uint l_i = 0; l_i < 4; l_i++)
    {
        uint l_idx = l_ind + l_i;
        uint4 l_ind_out;
        uint l_temp;

        l_temp = l_idx / l_dim_input[3];
        l_ind_out[3] = l_idx - l_temp * l_dim_input[3];
        l_idx = l_temp;

        l_temp = l_idx / l_dim_input[2];
        l_ind_out[2] = l_idx - l_temp * l_dim_input[2];
        l_idx = l_temp;

        l_temp = l_idx / l_dim_input[1];
        l_ind_out[1] = l_idx - l_temp * l_dim_input[1];
        l_idx = l_temp;

        l_ind_out[0] = l_idx;

        l_temp = l_ind_out[l_axes[3]] + l_dim_output[3] * (l_ind_out[l_axes[2]] + l_dim_output[2] * (l_ind_out[l_axes[1]] + l_dim_output[1] * l_ind_out[l_axes[0]]));

        l_output[l_temp] = l_input4[l_i];
    }
    GroupMemoryBarrierWithGroupSync();

    l_input4 = uint4(l_output[l_ind], l_output[l_ind + 1], l_output[l_ind + 2], l_output[l_ind + 3]);
    l_tensors.Store4(l_tensor_out_offset + 4 * l_ind, l_input4);
}
