// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 5
// uint m_tensor_offset_0;
// uint m_tensor_offset_1;
// uint m_tensor_offset_2;
// uint m_tensor_offset_3;
// uint m_tensor_offset_4;

// UP TO 15 TENSORS CAN BE CONCATENATED
#define MAX_TENSORS 16
#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    uint l_byte_offset_tensor[MAX_TENSORS] =
    {
        l_meta_data.m_tensor_offset_0, l_meta_data.m_tensor_offset_1,
        l_meta_data.m_tensor_offset_2, l_meta_data.m_tensor_offset_3,
        l_meta_data.m_tensor_offset_4, l_meta_data.m_tensor_offset_5,
        l_meta_data.m_tensor_offset_6, l_meta_data.m_tensor_offset_7,
        l_meta_data.m_tensor_offset_8, l_meta_data.m_tensor_offset_9,
        l_meta_data.m_tensor_offset_10, l_meta_data.m_tensor_offset_11,
        l_meta_data.m_tensor_offset_12, l_meta_data.m_tensor_offset_13,
        l_meta_data.m_tensor_offset_14, l_meta_data.m_tensor_offset_15
    };
    uint l_size_tensor[MAX_TENSORS];

    // Save byte offsets, shapes, and flattened sizes
    for (uint l_i = 0; l_i < l_meta_data.m_tensor_count; l_i++)
    {
        uint4 l_shape_tensor;

        l_byte_offset_tensor[l_i] += tensor_shape(l_tensors, l_byte_offset_tensor[l_i], l_shape_tensor);
        l_size_tensor[l_i] = shape_size(l_shape_tensor);
    }


    // flattened output index of particular thread
    uint l_idx = GROUP_SIZE * GROUP_SIZE * p_gid.x + p_gi;

    // m_tensor_count-1 is the output tensor index
    if (l_idx < l_size_tensor[l_meta_data.m_tensor_count-1])
    {
        float l_out = 0;
        uint l_boundary = 0;

        // from looking at the output index (l_idx), determine which of the input tensor to go to
        // then obtain appropriate element from that input tensor
        for (uint l_bin = 0; l_bin < l_meta_data.m_tensor_count - 1; l_bin++)
        {
            l_boundary += l_size_tensor[l_bin];

            if (l_idx < l_boundary)
            {
                l_byte_offset_tensor[l_bin] += 4 * (l_idx - l_boundary + l_size_tensor[l_bin]);
                l_out = asfloat(l_tensors.Load(l_byte_offset_tensor[l_bin]));
                break;
            }
        }

        // Store result
        l_byte_offset_tensor[l_meta_data.m_tensor_count - 1] += 4 * l_idx;
        l_tensors.Store(l_byte_offset_tensor[l_meta_data.m_tensor_count - 1], asuint(l_out));
    }
}
