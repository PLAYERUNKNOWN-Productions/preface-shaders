// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // input
// uint m_tensor_offset_1; // mat2
// uint m_tensor_offset_2; // output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint4 l_buffer = asuint(l_tensors.Load4(l_meta_data.m_tensor_offset_2));

    uint3 l_sh_output = uint3(l_buffer[1], l_buffer[2], l_buffer[3]);
    uint l_n_col_input = asuint(l_tensors.Load(l_meta_data.m_tensor_offset_0 + 12));

    // Get location of data
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 4 * (1 + 3);
    uint l_in_byte_offset_mat2 = l_meta_data.m_tensor_offset_1 + 4 * (1 + 3);
    uint l_out_byte_offset_output = l_meta_data.m_tensor_offset_2 + 4 * (1 + 3);

    uint l_id1 = p_gid.z;
    uint l_id2 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_id3 = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_id2 < l_sh_output[1] && l_id3 < l_sh_output[2])
    {   
        float l_sum = 0;
        uint l_id_output = l_id1 * l_sh_output[1] * l_sh_output[2] + l_id2 * l_sh_output[2] + l_id3;

        for (uint l_i = 0; l_i < l_n_col_input; l_i++)
        {
            uint l_id_input = l_i + l_n_col_input * (l_id2 + l_sh_output[1] * l_id1);
            uint l_id_mat2 = l_id3 + l_sh_output[2] * (l_i + l_n_col_input * l_id1);

            float l_element_a = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * l_id_input));
            float l_element_b = asfloat(l_tensors.Load(l_in_byte_offset_mat2 + 4 * l_id_mat2));

            l_sum += l_element_a * l_element_b;
        }

        // Store result
        l_tensors.Store(l_out_byte_offset_output + 4 * l_id_output, asuint(l_sum));
    }
}
