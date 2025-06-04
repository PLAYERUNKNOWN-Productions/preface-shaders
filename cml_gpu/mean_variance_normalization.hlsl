// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;     // 4
// uint m_tensor_offset_0;  // input
// uint m_tensor_offset_1;  // scale
// uint m_tensor_offset_2;  // offset
// uint m_tensor_offset_3;  // output

#define GROUP_SIZE 16

groupshared float l_input[4096];
groupshared float l_scale[512];
groupshared float l_offset[512];

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Rank 4 assumed
    uint4 l_shape = asuint(l_tensors.Load4(l_meta_data.m_tensor_offset_0 + 4));

    uint l_byte_offset_input = l_meta_data.m_tensor_offset_0 + 20;
    uint l_byte_offset_scale = l_meta_data.m_tensor_offset_1 + 12;      // Assume [1, l_shape[1]]
    uint l_byte_offset_offset = l_meta_data.m_tensor_offset_2 + 12;     // [1, l_shape[1]]
    uint l_byte_offset_output = l_meta_data.m_tensor_offset_3 + 20;     // Same dimension as input

    // Retrieve reduction l_axes
    uint l_axes[4] = {0, 0, 0, 0};

    for (uint l_ii = 0; l_ii < l_meta_data.m_attrib_count - 1; l_ii++)
    {
        uint l_axis_dim = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (1 + l_ii)));
        l_axes[l_axis_dim] = 1;
    }

    // epsilon
    float l_attrib_epsilon = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * l_meta_data.m_attrib_count));

    uint l_y = p_gid.z;
    uint l_z = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_w = p_gid.x * GROUP_SIZE + p_gtid.x;

    uint l_increment_2 = l_shape[3];

    uint l_idx_end_2 = l_axes[2] ? l_shape[2] : 1;
    uint l_idx_end_3 = l_axes[3] ? l_shape[3] : 1;

    uint l_id_mean = (1 - l_axes[3]) * l_w + l_shape[3] * ((1 - l_axes[2]) * l_z + l_shape[2] * (1 - l_axes[1]) * l_y);
    uint l_id = l_w + l_shape[3] * (l_z + l_shape[2] * l_y);

    uint l_ch_offset = l_y * l_shape[2] * l_shape[3];

    // Copy last two dimensions of input data to shared memory
    for (uint l_temp_id = p_gtid.y * 16 + p_gtid.x; l_temp_id < 4096; l_temp_id += 256)
    {
        l_input[l_temp_id] = asfloat(l_tensors.Load(l_byte_offset_input + 4 * (l_temp_id + l_ch_offset)));
    }

    // Copy scale and offset vectors(?) to shared memory
    for (uint l_temp_id = p_gtid.y * 16 + p_gtid.x; l_temp_id < 512; l_temp_id += 256)
    {
        l_scale[l_temp_id] = asfloat(l_tensors.Load(l_byte_offset_scale + 4 * l_temp_id));
        l_offset[l_temp_id] = asfloat(l_tensors.Load(l_byte_offset_offset + 4 * l_temp_id));
    }

    GroupMemoryBarrierWithGroupSync();

    // Total number of elements to be averaged
    float l_total = (float)(l_idx_end_2 * l_idx_end_3);
    float l_var, l_mean;

    if (l_z < l_shape[2] && l_w < l_shape[3])
    {
        float l_sum = 0;

        // Compute mean for channel y
        for (uint l_j = 0; l_j < l_idx_end_2; l_j++)
        {
            for (uint l_k = 0; l_k < l_idx_end_3; l_k++)
            {
                uint l_idx = l_id_mean + l_j * l_increment_2 + l_k;
                l_sum += l_input[l_idx - l_ch_offset];
            }
        }
        l_mean = l_sum / l_total;
        l_sum = 0;

        // Compute variance for channel y
        for (uint l_k = 0; l_k < l_idx_end_2; l_k++)
        {
            for (uint l_l = 0; l_l < l_idx_end_3; l_l++)
            {
                uint l_idx = l_id_mean + + l_k * l_increment_2 + l_l;
                float l_value = l_input[l_idx - l_ch_offset] - l_mean;
                l_sum += l_value * l_value;
            }
        }
        l_var = l_sum / l_total;

        float l_scale1 = l_scale[l_y];
        float l_offset2 = l_offset[l_y];
        float l_input1 = l_input[l_id - l_ch_offset];

        float l_value = l_scale1 * (l_input1 - l_mean) / sqrt(l_var + l_attrib_epsilon) + l_offset2;

        // Store result
        l_tensors.Store(l_byte_offset_output + 4 * l_id, asuint(l_value));
    }
}
