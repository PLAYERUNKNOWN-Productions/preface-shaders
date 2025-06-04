// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 3
// uint m_tensor_offset_0; // Input a
// uint m_tensor_offset_1; // Input b
// uint m_tensor_offset_2; // Output

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    if (l_meta_data.m_attrib_count != 1)
    {
        CML_SET_ERROR_INT(0, l_meta_data.m_attrib_count);
        CML_SET_KERNEL_ERROR;
    }

    // Up to rank 6 tensors are allowed
    // all three tensors have the same shapes!
    uint l_dim[6]; // shape
    uint l_in_rank_a = asuint(l_tensors.Load(int(l_meta_data.m_tensor_offset_0)));
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;

    // get shape.. unsqueeze to rank 6
    for (uint l_i = 0; l_i < 6; l_i++)
    {
        l_dim[l_i] = 1;
    }
    for (uint l_j = 0; l_j < l_in_rank_a; l_j++)
    {
        l_dim[l_j] = asuint(l_tensors.Load(l_in_byte_offset_a + 4 * (1 + l_j)));
    }

    l_in_byte_offset_a += 4 * (1 + l_in_rank_a);
    uint l_in_byte_offset_b = l_meta_data.m_tensor_offset_1 + 4 * (1 + l_in_rank_a);
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_2 + 4 * (1 + l_in_rank_a);

    // axis where x,y,z coordinates are located
    uint l_axis = asuint(l_attributes.Load(l_meta_data.m_attrib_offset + 4));

    // flattened increments for "axis"
    uint l_inc = 1;
    for (uint l_k = l_axis + 1; l_k < 6; l_k++)
    {
        l_inc *= l_dim[l_k];
    }

    // total number of pairs of vectors x 3 (components)
    uint l_n = l_dim[0] * l_dim[1] * l_dim[2] * l_dim[3] * l_dim[4] * l_dim[5];

    // flattened index for this particular thread
    uint l_id = p_gid.x * GROUP_SIZE * GROUP_SIZE + p_gtid.y * GROUP_SIZE + p_gtid.x;

    if (l_id < l_n)
    {
        uint l_remain = l_id;
        uint l_ind[6];

        for (uint l_i = 5; l_i > 0; l_i--)
        {
            l_ind[l_i] = l_remain % l_dim[l_i];
            l_remain = (l_remain - l_ind[l_i]) / l_dim[l_i];
        }
        l_ind[0] = l_remain;

        // offsets for x, y, z depending on the output component
        uint l_n_0 = (l_ind[l_axis] == 0) * 0 + (l_ind[l_axis] == 1) * (-1) + (l_ind[l_axis] == 2) * (-2);
        uint l_n_1 = (l_ind[l_axis] == 0) * 1 + (l_ind[l_axis] == 1) * 0    + (l_ind[l_axis] == 2) * (-1);
        uint l_n_2 = (l_ind[l_axis] == 0) * 2 + (l_ind[l_axis] == 1) * 1    + (l_ind[l_axis] == 2) * 0;

        l_n_0 *= l_inc;
        l_n_1 *= l_inc;
        l_n_2 *= l_inc;

        // first vector components
        float l_a0 = asfloat(l_tensors.Load(int(l_in_byte_offset_a + 4 * (l_id + l_n_0))));
        float l_a1 = asfloat(l_tensors.Load(int(l_in_byte_offset_a + 4 * (l_id + l_n_1))));
        float l_a2 = asfloat(l_tensors.Load(int(l_in_byte_offset_a + 4 * (l_id + l_n_2))));

        // second vector components
        float l_b0 = asfloat(l_tensors.Load(int(l_in_byte_offset_b + 4 * (l_id + l_n_0))));
        float l_b1 = asfloat(l_tensors.Load(int(l_in_byte_offset_b + 4 * (l_id + l_n_1))));
        float l_b2 = asfloat(l_tensors.Load(int(l_in_byte_offset_b + 4 * (l_id + l_n_2))));

        // save right component to "product"
        float l_product = (l_ind[l_axis] == 0) * (l_a1 * l_b2 - l_a2 * l_b1)
                        + (l_ind[l_axis] == 1) * (l_a2 * l_b0 - l_a0 * l_b2)
                        + (l_ind[l_axis] == 2) * (l_a0 * l_b1 - l_a1 * l_b0);

        // Store result
        l_tensors.Store(l_out_byte_offset + 4 * l_id, asuint(l_product));
    }
}
