// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2 
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

groupshared uint l_output[4096];

#define GROUP_SIZE 1024
// if second element of output shape is even, N_LOOP = 2
// this makes things faster
// this works in coordination with transpose_a.inl
#define N_LOOP 2

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
    l_offset += 16; // dimensions already read

    // Each loop processes 4 elements
    // l_j up to 2 means each thread takes care of 4 x 2 elements
    for (uint l_j = 0; l_j < N_LOOP; l_j++)
    {
        uint l_ii = N_LOOP * p_gid.x + l_j;
    
        uint l_offset2 = l_offset + 4 * l_ii * l_n_output;

        // treat as rank 4 tensor
        uint l_tensor_in_offset = l_meta_data.m_tensor_offset_0 + l_offset2;
        uint l_tensor_out_offset = l_meta_data.m_tensor_offset_1 + l_offset2;

        // each thread takes care of 4 elements
        uint l_ind = 4 * p_gtid.x;
        uint4 l_input4 = l_tensors.Load4(l_tensor_in_offset + 4 * l_ind);

    #if 0 // permutation fixed (0,1,4,2,5,3); does not follow from attributes
        uint4 l_ind4 = uint4(l_ind, l_ind+1, l_ind+2, l_ind+3);
        uint4 l_ind_out0, l_ind_out1, l_ind_out2, l_ind_out3;
    
        uint4 l_temp4;

        l_temp4 = l_ind4 / l_dim_input[3];
        l_ind_out3 = l_ind4 - l_temp4 * l_dim_input[3];
        l_ind4 = l_temp4;

        l_temp4 = l_ind4 / l_dim_input[2];
        l_ind_out2 = l_ind4 - l_temp4 * l_dim_input[2];
        l_ind4 = l_temp4;

        l_temp4 = l_ind4 / l_dim_input[1];
        l_ind_out1 = l_ind4 - l_temp4 * l_dim_input[1];

        l_ind_out0 = l_temp4;

        l_temp4 = l_ind_out1 + l_dim_output[3] * (l_ind_out3 + l_dim_output[2] * (l_ind_out0 + l_dim_output[1] * l_ind_out2));
    
        l_output[l_temp4.x] = l_input4.x;
        l_output[l_temp4.y] = l_input4.y;
        l_output[l_temp4.z] = l_input4.z;
        l_output[l_temp4.w] = l_input4.w;
    #else
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
    #endif 
        GroupMemoryBarrierWithGroupSync();

        l_input4 = uint4(l_output[l_ind], l_output[l_ind+1], l_output[l_ind+2], l_output[l_ind+3]);
        l_tensors.Store4(l_tensor_out_offset + 4 * l_ind, l_input4);
    }
}