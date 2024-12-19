// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

#include "softmax_a_func.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define N_TENSORS 2
//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------

#define GROUP_SIZE 1
#define Z_SIZE 32
groupshared float l_sum[Z_SIZE*GROUP_SIZE*GROUP_SIZE];

[numthreads(Z_SIZE, GROUP_SIZE, GROUP_SIZE)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    uint4 l_shape; // shape same for both input and output tensors
    uint l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1 };

    l_shape = asuint(l_tensors.Load4(l_byte_offset_tensor[0] + 4));

    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {     
        l_byte_offset_tensor[l_i] += 4 * (1 + 4);
    }

    uint l_idx_x = p_gid.x;
    uint l_idx_y = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_idx_z = p_gid.z * GROUP_SIZE + p_gtid.z;

    // ASSUME REDUCTION AXIS = 3
    const uint l_idx_start = l_idx_z * l_shape[3] * l_shape[2] + l_idx_y * l_shape[3];
    #define l_axis 3

    uint l_id0 = Z_SIZE *(p_gtid.y + GROUP_SIZE * p_gtid.z);
    uint l_idx = l_idx_start;

    if (l_idx_y < l_shape[2] && l_idx_z < l_shape[1])
    {
        float l_temp = 0.0f;

        for (uint l_i = p_gtid.x; l_i < l_shape[l_axis]; l_i += Z_SIZE)
        {
            float value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx));
            l_temp += exp(value);
            l_idx += Z_SIZE;
        }
        l_sum[l_id0 + p_gtid.x] = l_temp;
    }
    GroupMemoryBarrierWithGroupSync();

    if (l_idx_y < l_shape[2] && l_idx_z < l_shape[1] && p_gtid.x == 0)
    {
        float l_total = 0.0f;
        for (uint l_i = 0; l_i < Z_SIZE; l_i++)
        {
            l_total += l_sum[l_id0 + l_i];
        }

        l_idx = l_idx_start + l_idx_x;
        float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx));
        float l_out = exp(l_value) / l_total;

        l_tensors.Store(l_byte_offset_tensor[1] + 4 * l_idx, asuint(l_out));
    }
}
