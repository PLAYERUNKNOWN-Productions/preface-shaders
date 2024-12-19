// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define GROUP_SIZE 1024
#define N_TENSORS 2
#define ID_OUT_TENSOR 1

struct randnormlike_attribs_s
{
    int dtype;
    float mean;
    float stdev;
    int seed;
};

float randnormlike_func(in float p_mean, in float p_stdev, in uint p_seed, in uint p_id)
{
    int l_id1 = p_seed + (int)(2 * p_id);
    int l_id2 = l_id1 + 1;

    uint l_a1 = (l_id1 * 15485863);
    uint l_a2 = (l_id2 * 15485863);

    l_a1 = l_a1 * l_a1 * l_a1;
    l_a2 = l_a2 * l_a2 * l_a2;

    float l_r1 = ((float)l_a1) / 4294967295.0f;
    float l_r2 = ((float)l_a2) / 4294967295.0f;

    if (l_r1 < 0.0001)
        l_r1 += 0.0001;

    return (float)(sqrt(-2.0 * log(l_r1)) * sin(2.0 * 3.14159265359 * l_r2) * p_stdev + p_mean);
}

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out randnormlike_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];
    // Check attribute count
    if (l_count != 4)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif  // PP_KERNEL_ERROR_HANDLING

    p_attribs.dtype = asuint(p_attrib_buffer.Load(l_byte_offset));
    l_byte_offset += 4;
    p_attribs.mean = asfloat(p_attrib_buffer.Load(l_byte_offset));
    l_byte_offset += 4;
    p_attribs.stdev = asfloat(p_attrib_buffer.Load(l_byte_offset));
    l_byte_offset += 4;
    p_attribs.seed = asuint(p_attrib_buffer.Load(l_byte_offset));
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
    uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    randnormlike_attribs_s l_attribs;
    get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    uint l_rank[N_TENSORS];
    uint l_shape[N_TENSORS][MB_CML_GPU_MAX_TENSOR_RANK];
    uint l_byte_offset_tensor[N_TENSORS] = { l_meta_data.m_tensor_offset_0,
                                               l_meta_data.m_tensor_offset_1 };

    // Go through every tensor to obtain shape/rank
    for (uint l_i = 0; l_i < N_TENSORS; l_i++)
    {
        l_rank[l_i] = asuint(l_tensors.Load(l_byte_offset_tensor[l_i]));
        get_tensor_shape(l_tensors, l_byte_offset_tensor[l_i], l_rank[l_i], l_shape[l_i]);
        l_byte_offset_tensor[l_i] += FLOAT_SIZE * (1 + l_rank[l_i]);
    }
    uint l_n_outputs = get_shape_size(l_shape[ID_OUT_TENSOR], l_rank[ID_OUT_TENSOR]);

    uint l_seed = asuint(l_scratch.Load(INT_SIZE * p_gid.x));
    uint l_idx_output = p_gtid.x + p_gid.x * GROUP_SIZE;

    if (l_idx_output < l_n_outputs)
    {
        float l_out = randnormlike_func(l_attribs.mean, l_attribs.stdev, l_seed + l_attribs.seed, l_idx_output);
        l_tensors.Store(l_byte_offset_tensor[ID_OUT_TENSOR] + FLOAT_SIZE * l_idx_output, asuint(l_out));

        if (p_gtid.x == 0)
        {
            l_scratch.Store(INT_SIZE * p_gid.x, asuint(l_seed + 2 * l_n_outputs));
        }
    }
}
