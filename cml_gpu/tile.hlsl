// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 256
#define N_TENSORS 2
#define ID_OUT_TENSOR 1


void get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out uint2 p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count > 2)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return;
    }
#endif

    if (l_count == 1)
    {
        p_attribs[0] = p_attrib_buffer.Load(l_byte_offset);
        p_attribs[1] = 1;
    }
    else // if (l_count == 2)
    {
        p_attribs = p_attrib_buffer.Load2(l_byte_offset);
    }
}


[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    if (l_meta_data.m_attrib_count != 0)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }

    uint    l_rank[N_TENSORS];
    uint    l_shape[N_TENSORS][MB_CML_GPU_MAX_TENSOR_RANK];
    uint    l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0,
                                                  l_meta_data.m_tensor_offset_1 };
    uint    l_n_outputs = 1;

    // Obtain ranks and shapes of all tensors
    // Then adjust offset to 0th element of tensor
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));

        for (uint l_j = 0; l_j < l_rank[l_i]; l_j++)
        {
            l_shape[l_i][l_j] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i] + INT_SIZE * (1 + l_j)));
        }
        for (uint l_j = l_rank[l_i]; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
        {
            l_shape[l_i][l_j] = 1;
        }

        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }

    // number of outputs
    for (uint l_i = 0; l_i < l_rank[ID_OUT_TENSOR]; l_i++)
    {
        l_n_outputs *= l_shape[ID_OUT_TENSOR][l_i];
    }

    uint l_idx_out = p_gid.x * GROUP_SIZE + p_gtid.x;

    if (l_idx_out < l_n_outputs)
    {
        uint l_output_idx[MB_CML_GPU_MAX_TENSOR_RANK];
        uint l_input_idx[MB_CML_GPU_MAX_TENSOR_RANK];

        out_flatten_to_coord(l_idx_out, l_shape[ID_OUT_TENSOR], l_rank[ID_OUT_TENSOR], l_output_idx);

        for (uint l_i = 0; l_i < l_rank[0]; l_i++)
        {
            l_input_idx[l_i] = l_output_idx[l_i] % l_shape[0][l_i];
        }

        uint l_idx_in = (uint)in_coord_to_flatten(l_input_idx, l_shape[0], l_rank[0]);
        float l_value = asfloat(l_tensors.Load(l_byte_offset_tensor[0] + FLOAT_SIZE * l_idx_in));

        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_out, asuint(l_value));
    }
}
