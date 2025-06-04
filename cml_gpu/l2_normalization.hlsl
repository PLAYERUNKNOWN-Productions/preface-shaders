// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0;
// uint m_tensor_offset_1;

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint4 l_shape_tensor[2];
    uint l_byte_offset_tensor[] = { l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1 };
    uint l_size_tensor[2];
    uint l_axes[4];

    // Save byte offsets, shapes, and flattened sizes
    for (uint l_i = 0; l_i < l_meta_data.m_tensor_count; l_i++)
    {
        l_byte_offset_tensor[l_i] += tensor_shape(l_tensors, l_byte_offset_tensor[l_i], l_shape_tensor[l_i]);
        l_size_tensor[l_i] = shape_size(l_shape_tensor[l_i]);
    }

    uint l_reduce_axis = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4));
    float l_bias = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 8));
    float l_epsilon = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 12));

    for (uint l_j = 0; l_j < 4; l_j++)
    {
        if (l_j == l_reduce_axis)
        {
            l_axes[l_j] = 1;
        }
        else
        {
            l_axes[l_j] = 0;
        }
    }

    uint l_y = p_gid.z;
    uint l_z = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_w = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_y < l_shape_tensor[1].y && l_z < l_shape_tensor[1].z && l_w < l_shape_tensor[1].w)
    {
        int l_increment = l_axes[1] * l_shape_tensor[0].w * l_shape_tensor[0].z + l_axes[2] * l_shape_tensor[0].w + l_axes[3];
        int l_idx0 = (1 - l_axes[1]) * l_y * l_shape_tensor[0].w * l_shape_tensor[0].z + (1 - l_axes[2]) * l_z * l_shape_tensor[0].w + (1 - l_axes[3]) * l_w;
        int l_idx_end = l_axes[1] * l_shape_tensor[0].y + l_axes[2] * l_shape_tensor[0].z + l_axes[3] * l_shape_tensor[0].w;

        float l_max_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx0));
        float l_sum = 0;

        for (uint l_i = 0; l_i < l_idx_end; l_i++)
        {
            uint l_idx = l_idx0 + l_i * l_increment;
            float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx));

            l_sum += l_value * l_value;
        }

        int l_idx_out = l_y * l_shape_tensor[1].w * l_shape_tensor[1].z + l_z * l_shape_tensor[1].w + l_w;
        float l_out = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + 4 * l_idx_out));

        l_tensors.Store(l_byte_offset_tensor[1] + 4 * l_idx_out, asuint(l_out / max(sqrt(l_sum) + l_bias, l_epsilon)));
    }
}
