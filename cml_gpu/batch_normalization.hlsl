// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

#include "batch_normalization_func.hlsl"

// uint m_tensor_count;    // 6
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Mean
// uint m_tensor_offset_2; // Variance
// uint m_tensor_offset_3; // Offset
// uint m_tensor_offset_4; // Scale
// uint m_tensor_offset_5; // Output

#define GROUP_SIZE 16
#define N_TENSORS 6
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_rank[N_TENSORS];
    uint4 l_shape[N_TENSORS];
    uint l_byte_offset_tensor[N_TENSORS] =
    {
            l_meta_data.m_tensor_offset_0,
            l_meta_data.m_tensor_offset_1,
            l_meta_data.m_tensor_offset_2,
            l_meta_data.m_tensor_offset_3,
            l_meta_data.m_tensor_offset_4,
            l_meta_data.m_tensor_offset_5
    };

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        l_shape[l_i].x = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 0)));
        l_shape[l_i].y = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 1)));

        if (l_rank[l_i] == 4)
        {
            l_shape[l_i].z = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 2)));
            l_shape[l_i].w = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + 4 * (1 + 3)));
        }

        l_byte_offset_tensor[l_i] += 4 * (1 + l_rank[l_i]);
    }

    // Obtain attributes
    float l_epsilon = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 4));

    uint l_k2 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_k2 < l_shape[5].y && l_i1 < l_shape[5].z && l_i2 < l_shape[5].w)
    {
        uint l_idx_out = l_k2 * l_shape[5].z * l_shape[5].w + l_i1 * l_shape[5].w + l_i2;

        float l_input = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx_out));
        float l_mean = asfloat(l_tensors.Load(l_byte_offset_tensor[1] + 4 * l_k2));
        float l_variance = asfloat(l_tensors.Load(l_byte_offset_tensor[2] + 4 * l_k2));
        float l_offset = asfloat(l_tensors.Load(l_byte_offset_tensor[3] + 4 * l_k2));
        float l_scale = asfloat(l_tensors.Load(l_byte_offset_tensor[4] + 4 * l_k2));

        float l_output;

        l_output = l_input - l_mean;
        l_output *= l_scale / sqrt(l_variance + l_epsilon);
        l_output += l_offset;

        l_tensors.Store(l_byte_offset_tensor[5] + 4 * l_idx_out, asuint(l_output));
    }
}
