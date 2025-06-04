// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Index
// uint m_tensor_offset_2; // Output

#define GROUP_SIZE 16
#define N_TENSORS 3

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_rank[N_TENSORS];
    uint l_shape[N_TENSORS][4];
    uint l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1, l_meta_data.m_tensor_offset_2 };

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for(uint l_j = 0; l_j < 4; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }

        for(uint l_jj = 0; l_jj < l_rank[l_i]; l_jj++)
        {
            l_shape[l_i][l_jj] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + l_jj)));
        }

        l_byte_offset_tensor[l_i] += 4 * (1 + l_rank[l_i]);
    }

    // Obtain attributes
    uint l_axes_index = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4));

    uint l_axes[4] = { 0,0,0,0 };
    l_axes[l_axes_index] = 1;

    uint l_y = p_gid.z;
    uint l_z = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_w = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_y < l_shape[2][1] && l_z < l_shape[2][2] && l_w < l_shape[2][3])
    {
        uint l_index = l_axes[1] * asuint(l_tensors.Load(l_byte_offset_tensor[1] + 4 * l_y))
                     + l_axes[2] * asuint(l_tensors.Load(l_byte_offset_tensor[1] + 4 * l_z))
                     + l_axes[3] * asuint(l_tensors.Load(l_byte_offset_tensor[1] + 4 * l_w));

        uint l_idx0 = (1 - l_axes[1]) * l_y * l_shape[0][3] * l_shape[0][2] + (1 - l_axes[2]) * l_z * l_shape[0][3] + (1 - l_axes[3]) * l_w;
        l_idx0 += l_axes[1] * l_index * l_shape[0][3] * l_shape[0][2] + l_axes[2] * l_index * l_shape[0][3] + l_axes[3] * l_index;

        uint l_idx_out = l_y * l_shape[2][3] * l_shape[2][2] + l_z * l_shape[2][3] + l_w;

        float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx0));

        l_tensors.Store(l_byte_offset_tensor[2] + 4 * l_idx_out, asuint(l_value));
    }
}
