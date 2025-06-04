// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0;
// uint m_tensor_offset_1;

#define GROUP_SIZE 16
#define N_TENSORS 2

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_rank[N_TENSORS];
    uint l_shape[N_TENSORS][4];
    uint l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1 };

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        l_shape[l_i][0] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 0)));
        l_shape[l_i][1] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 1)));
        l_shape[l_i][2] = 1;
        l_shape[l_i][3] = 1;

        if (l_rank[l_i] == 4)
        {
            l_shape[l_i][2] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 2)));
            l_shape[l_i][3] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 3)));
        }

        l_byte_offset_tensor[l_i] += 4 * (1 + l_rank[l_i]);
    }

    // Retrieve reduction l_axes
    uint l_axes[4] = { 0,0,0,0 };

    for (uint l_ii = 0; l_ii < l_meta_data.m_attrib_count; l_ii++)
    {
        uint l_axis_dim = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (1 + l_ii)));
        l_axes[l_axis_dim] = 1;
    }

    uint l_y = p_gid.z;
    uint l_z = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_w = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_y < l_shape[1][1] && l_z < l_shape[1][2] && l_w < l_shape[1][3])
    {
        uint l_idx0 = (1 - l_axes[1]) * l_y * l_shape[0][3] * l_shape[0][2] + (1 - l_axes[2]) * l_z * l_shape[0][3] + (1 - l_axes[3]) * l_w;
        uint l_idx_out = l_y * l_shape[1][3] * l_shape[1][2] + l_z * l_shape[1][3] + l_w;

        uint l_increment_1 = l_shape[0][3] * l_shape[0][2];
        uint l_increment_2 = l_shape[0][3];

        uint l_idx_end_1 = l_axes[1] ? l_shape[0][1] : 1;
        uint l_idx_end_2 = l_axes[2] ? l_shape[0][2] : 1;
        uint l_idx_end_3 = l_axes[3] ? l_shape[0][3] : 1;

        float l_max_value = -FLT_MAX;

        for (uint l_i = 0; l_i < l_idx_end_1; l_i++)
        {
            for (uint l_j = 0; l_j < l_idx_end_2; l_j++)
            {
                for (uint l_k = 0; l_k < l_idx_end_3; l_k++)
                {
                    uint l_idx = l_idx0 + l_i * l_increment_1 + l_j * l_increment_2 + l_k;
                    float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx));

                    uint l_cond = l_value > l_max_value;
                    l_max_value = l_cond * l_value + (1 - l_cond) * l_max_value;
                }
            }
        }

        l_tensors.Store(l_byte_offset_tensor[1] + 4 * l_idx_out, asuint(l_max_value));
    }
}
