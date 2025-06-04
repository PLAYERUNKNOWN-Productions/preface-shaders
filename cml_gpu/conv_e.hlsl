// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 7
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Filter
// uint m_tensor_offset_2; // Bias
// uint m_tensor_offset_3; // Output
// uint m_tensor_offset_4; // Matrix Buffer

#define GROUP_SIZE 32

#define l_d_o 32
#define l_d_i 32
#define l_out_ch 192
#define l_in_ch 48
#define l_d_f 3
#define l_s1 1
#define l_s2 1
#define l_d1 1
#define l_d2 1
#define l_p1 1
#define l_p2 1

#define l_o1 (p_gid.x)
#define l_o2 (p_gtid.y)
#define l_o3 (p_gtid.x)

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1 + 20;
    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2 + 12;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3 + 20;
    uint l_temp_matrix_offset = l_meta_data.m_tensor_offset_4 + 12;

    uint l_df_sqr = l_d_f * l_d_f;
    uint l_filter_size = l_in_ch * l_df_sqr;
    uint l_idx_output = l_o1 * l_d_o * l_d_o + l_o2 * l_d_o + l_o3;

    uint l_idx_filter0 = l_o1 * l_filter_size; // offset for current l_o1
    uint l_idx_matrix0 = (l_o2 * l_d_o + l_o3) * l_filter_size; // offset for current (l_o2,l_o3)

    // 1. unroll input tensor to matrix, including padding
    for (uint l_c = 0; l_c < l_in_ch; l_c++)
    {
        // offsets for current in-channel
        uint l_idx_input0 = l_c * l_d_i * l_d_i;

        for (uint l_i = 0; l_i < l_d_f; l_i++)
        {
            uint l_ind1 = l_o2 + l_i - l_p1;

            for (uint l_j = 0; l_j < l_d_f; l_j++)
            {
                uint l_ind2 = l_o3 + l_j - l_p1;

                float l_t1 = 0.0f;

                // load input pixel corresponding to filter (l_i, l_j)
                if (l_ind1 >= 0 && l_ind1 < l_d_i && l_ind2 >= 0 && l_ind2 < l_d_i)
                {
                    uint l_idx_input = l_idx_input0 + l_ind2 + l_ind1 * l_d_i;
                    l_t1 = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * l_idx_input));
                }

                uint l_idx_matrix = l_idx_matrix0 + (l_c * l_df_sqr + l_i * l_d_f + l_j);
                l_tensors.Store(l_temp_matrix_offset + 4 * l_idx_matrix, asuint(l_t1));
            }
        }
    }

    // 2. matrix multiplication W * input
    #define INC 64
    for (uint l_z = l_o1; l_z < 192; l_z += INC)
    {
        l_idx_output = l_z * l_d_o * l_d_o + l_o2 * l_d_o + l_o3;
        l_idx_filter0 = l_z * l_filter_size; // offset for current l_o1

        float l_temp = 0.0f;

        for (uint l_i = 0; l_i < l_filter_size; l_i += 4)
        {
            float4 l_t1 = asfloat(l_tensors.Load4(l_temp_matrix_offset + 4 * (l_idx_matrix0 + l_i)));
            float4 l_t2 = asfloat(l_tensors.Load4(l_in_byte_offset_filter + 4 * (l_idx_filter0 + l_i)));

            l_temp += dot(l_t1, l_t2);
        }

        uint l_bias_address = l_in_byte_offset_bias + 4 * l_z;

        float l_out = l_temp + asfloat(l_tensors.Load(l_bias_address));
        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_out));
    }
}
