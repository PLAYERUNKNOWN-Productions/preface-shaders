// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 42
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Filter
// uint m_tensor_offset_2; // Bias
// uint m_tensor_offset_3; // Output 2

// Attributes specifically for this operation
struct conv_b_attribs_s
{
    int4 m_padding;
    int2 m_stride;
    int2 m_dilation;
    int m_groups;
};

uint get_attribs(in ByteAddressBuffer p_attrib_buffer, in uint p_byte_offset, out conv_b_attribs_s p_attribs)
{
    uint l_byte_offset = p_byte_offset;
    uint l_count = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

#ifdef CML_KERNEL_ERROR_HANDLING
    RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];

    // Check attribute count
    if (l_count != 9)
    {
        l_error_buffer.Store(4, l_count);
        l_error_buffer.Store(8, p_byte_offset);
        CML_SET_KERNEL_ERROR;
        return 0;
    }
#endif // CML_KERNEL_ERROR_HANDLING

    p_attribs.m_padding = p_attrib_buffer.Load4(l_byte_offset);
    l_byte_offset += 16;
    p_attribs.m_stride = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_dilation = p_attrib_buffer.Load2(l_byte_offset);
    l_byte_offset += 8;
    p_attribs.m_groups = p_attrib_buffer.Load(l_byte_offset);
    l_byte_offset += 4;

    return (l_byte_offset - p_byte_offset);
}

#define l_d_o (l_out_shape[3])
#define l_d_i (l_in_shape_input[3])
#define l_out_ch (l_out_shape[1])
#define l_in_ch (l_in_shape_input[1])
#define l_d_f (l_in_shape_filter[2])
#define l_s1 (l_attribs.m_stride[0])
#define l_s2 (l_attribs.m_stride[1])
#define l_d1 (l_attribs.m_dilation[0])
#define l_d2 (l_attribs.m_dilation[1])
#define l_p1 (l_attribs.m_padding[0])
#define l_p2 (l_attribs.m_padding[2])

#define FLOAT_SIZE 4

// ASSUMPTIONS
// l_d_f <= 4
// l_d1 == l_d2 == 1
// l_attribs.m_padding[0] = l_attribs.m_padding[1]
// l_attribs.m_padding[2] = l_attribs.m_padding[3]
//


// assumptions
// stride 1, padding 1, kernel 3

// 128 x 3 x 3 = 1152
// 18 x 18 = 324

#define P_TILE 18
#define TILE 16
#define PAD_SIZE 1
#define D 3
#define D2 9

#define FILTER_SIZE 1152
#define PAD_INPUT_SIZE 324
#define N_FILTER 2
#define N_INPUT 2

groupshared float l_filter[N_FILTER][FILTER_SIZE];
groupshared float l_input[N_INPUT][PAD_INPUT_SIZE];

[numthreads(1,18,18)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
    uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    //   We cannot enable below when we have GroupMemoryBarrierWithGroupSync();
    //CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    //conv_a_attribs_s l_attribs;
    //get_attribs(l_attributes, l_meta_data.m_attrib_offset, l_attribs);

    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint4 l_in_shape_filter;
    uint l_in_byte_offset_filter = l_meta_data.m_tensor_offset_1;
    l_in_byte_offset_filter += tensor_shape(l_tensors, l_in_byte_offset_filter, l_in_shape_filter);

    uint l_in_byte_offset_bias = l_meta_data.m_tensor_offset_2;
    uint2 l_in_shape_bias = asuint(l_tensors.Load2(l_in_byte_offset_bias + FLOAT_SIZE));
    l_in_byte_offset_bias += FLOAT_SIZE * 3; // rank, dim(2)

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_3;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint l_i1 = p_gid.y * TILE + (p_gtid.y - PAD_SIZE);    // image size (0-255)
    uint l_i2 = p_gid.z * TILE + (p_gtid.z - PAD_SIZE);
    uint l_k2 = 2 * p_gid.x;   //two output channels at once

    uint l_tid = p_gtid.y * P_TILE + p_gtid.z;  // 0 to 255; 1D id within threadgroup
    uint cond1 = l_i1 >= 0 && l_i1 < l_in_shape_input[2] && l_i2 >= 0 && l_i2 < l_in_shape_input[3];
    uint cond2 = p_gtid.y >= PAD_SIZE && p_gtid.y < (P_TILE - PAD_SIZE)
        && p_gtid.z >= PAD_SIZE && p_gtid.z < (P_TILE - PAD_SIZE);
    uint cond3 = l_in_shape_bias[1] > 1;

    uint l_inc = FLOAT_SIZE * l_in_shape_input[2] * l_in_shape_input[3];
    float l_sum = 0.0;
    float l_sum2 = 0.0;
    uint filter_size = l_in_shape_filter[1] * D2;
    float l_addr = l_in_byte_offset_filter + FLOAT_SIZE * filter_size * l_k2;

    // read filter for channel l_k1 and copy to groupshared
    for (uint l_i = l_tid; l_i < filter_size; l_i += PAD_INPUT_SIZE)
    {
        l_filter[0][l_i] = asfloat(l_tensors.Load(l_addr + FLOAT_SIZE * l_i));

        if (l_k2 + 1 < l_out_shape[1])
            l_filter[1][l_i] = asfloat(l_tensors.Load(l_addr + FLOAT_SIZE* (filter_size + l_i)));
        else
            l_filter[1][l_i] = 0.0;
    }

    l_addr = l_in_byte_offset_input + FLOAT_SIZE * (l_i1 * l_in_shape_input[3] + l_i2);

    uint l_k1 = 0;
    uint end = l_in_shape_filter[1] - 1;

    for (; l_k1 < end; l_k1 += 2)
    {
        // get input cell (18x18)
        if (cond1)
        {
            l_input[0][l_tid] = asfloat(l_tensors.Load(l_addr));
            l_input[1][l_tid] = asfloat(l_tensors.Load(l_addr + l_inc));
        }
        else
        {
            l_input[0][l_tid] = 0.0;
            l_input[1][l_tid] = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // compute output if not halo cell
        if (cond2)
        {
            uint offset = l_k1 * D2;
            for (uint l_j1 = 0; l_j1 < D; l_j1++)
            {
                for (uint l_j2 = 0; l_j2 < D; l_j2++)
                {
                    uint temp = offset + l_j2;
                    uint temp2 = (p_gtid.y - 1 + l_j1) * P_TILE + p_gtid.z - 1 + l_j2;
                    l_sum += l_filter[0][temp] * l_input[0][temp2];
                    l_sum2 += l_filter[1][temp] * l_input[0][temp2];
                    l_sum += l_filter[0][temp + D2] * l_input[1][temp2];
                    l_sum2 += l_filter[1][temp + D2] * l_input[1][temp2];
                }
                offset += D;
            }
        }
        l_addr += 2 * l_inc;
        GroupMemoryBarrierWithGroupSync();
    }

    if (l_k1 == end)
    {
        // get input cell (18x18)
        if (cond1)
        {
            l_input[0][l_tid] = asfloat(l_tensors.Load(l_addr));
        }
        else
        {
            l_input[0][l_tid] = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        // compute output if not halo cell
        if (cond2)
        {
            uint offset = l_k1 * D2;
            for (uint l_j1 = 0; l_j1 < D; l_j1++)
            {
                for (uint l_j2 = 0; l_j2 < D; l_j2++)
                {
                    uint temp = offset + l_j2;
                    uint temp2 = (p_gtid.y - 1 + l_j1) * P_TILE + p_gtid.z - 1 + l_j2;
                    l_sum += l_filter[0][temp] * l_input[0][temp2];
                    l_sum2 += l_filter[1][temp] * l_input[0][temp2];
                }
                offset += D;
            }
        }
    }


    if (cond2)
    {
        // Get bias value
        // Do not add offset if shape_bias = [1, 1]
        float l_bias = 0.0;

        if (cond3)
        {
            l_in_byte_offset_bias += FLOAT_SIZE * l_k2;
        }
        l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));


        uint l_idx_output = l_out_shape[3] * l_i1 + l_i2 + l_k2 * l_out_shape[2] * l_out_shape[3];
        l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * l_idx_output, asuint(l_sum +l_bias  ));

        if (l_k2 + 1 < l_out_shape[1])
        {
            l_bias = 0;
            if (cond3)
            {
                l_in_byte_offset_bias += FLOAT_SIZE;

            }
            l_bias = asfloat(l_tensors.Load(l_in_byte_offset_bias));
            uint l_idx_output2 = l_idx_output + l_out_shape[2] * l_out_shape[3];
            l_tensors.Store(l_out_byte_offset + FLOAT_SIZE * l_idx_output2, asuint(l_sum2 + l_bias ));
        }
    }

}
