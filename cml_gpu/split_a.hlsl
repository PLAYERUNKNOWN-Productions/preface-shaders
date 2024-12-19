// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 
// uint m_tensor_offset_0; // Tensors
// uint m_tensor_offset_1;
// uint m_tensor_offset_2;
// uint m_tensor_offset_3;
// uint m_tensor_offset_4;
// uint m_tensor_offset_5;
// uint m_tensor_offset_6;
// uint m_tensor_offset_7;
// uint m_tensor_offset_8;
// uint m_tensor_offset_9;
// uint m_tensor_offset_10;
// uint m_tensor_offset_11;
// uint m_tensor_offset_12;
// uint m_tensor_offset_13;
// uint m_tensor_offset_14;
// uint m_tensor_offset_15;
// uint m_tensor_offset_16; // Tensors
// uint m_tensor_offset_17;
// uint m_tensor_offset_18;
// uint m_tensor_offset_19;
// uint m_tensor_offset_20;
// uint m_tensor_offset_21;
// uint m_tensor_offset_22;
// uint m_tensor_offset_23;
// uint m_tensor_offset_24;
// uint m_tensor_offset_25;
// uint m_tensor_offset_26;
// uint m_tensor_offset_27;
// uint m_tensor_offset_28;
// uint m_tensor_offset_29;
// uint m_tensor_offset_30;
// uint m_tensor_offset_31;

// UP TO 15 TENSORS CAN BE CONCATENATED
#define MAX_TENSORS 32

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------

#define GROUP_SIZE 32
#define NN 1  //p_gid.y max
groupshared uint l_byte_offset_tensor[MAX_TENSORS];

[numthreads(GROUP_SIZE, 1, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    if (p_gtid.x == 0)
    {
        l_byte_offset_tensor[0] = l_meta_data.m_tensor_offset_0;
        l_byte_offset_tensor[1] = l_meta_data.m_tensor_offset_1;
        l_byte_offset_tensor[2] = l_meta_data.m_tensor_offset_2;
        l_byte_offset_tensor[3] = l_meta_data.m_tensor_offset_3;
        l_byte_offset_tensor[4] = l_meta_data.m_tensor_offset_4;
        l_byte_offset_tensor[5] = l_meta_data.m_tensor_offset_5;
        l_byte_offset_tensor[6] = l_meta_data.m_tensor_offset_6;
        l_byte_offset_tensor[7] = l_meta_data.m_tensor_offset_7;
        l_byte_offset_tensor[8] = l_meta_data.m_tensor_offset_8;
        l_byte_offset_tensor[9] = l_meta_data.m_tensor_offset_9;
        l_byte_offset_tensor[10] = l_meta_data.m_tensor_offset_10;
        l_byte_offset_tensor[11] = l_meta_data.m_tensor_offset_11;
        l_byte_offset_tensor[12] = l_meta_data.m_tensor_offset_12;
        l_byte_offset_tensor[13] = l_meta_data.m_tensor_offset_13;
        l_byte_offset_tensor[14] = l_meta_data.m_tensor_offset_14;
        l_byte_offset_tensor[15] = l_meta_data.m_tensor_offset_15;
        l_byte_offset_tensor[16] = l_meta_data.m_tensor_offset_16;
        l_byte_offset_tensor[17] = l_meta_data.m_tensor_offset_17;
        l_byte_offset_tensor[18] = l_meta_data.m_tensor_offset_18;
        l_byte_offset_tensor[19] = l_meta_data.m_tensor_offset_19;
        l_byte_offset_tensor[20] = l_meta_data.m_tensor_offset_20;
        l_byte_offset_tensor[21] = l_meta_data.m_tensor_offset_21;
        l_byte_offset_tensor[22] = l_meta_data.m_tensor_offset_22;
        l_byte_offset_tensor[23] = l_meta_data.m_tensor_offset_23;
        l_byte_offset_tensor[24] = l_meta_data.m_tensor_offset_24;
        l_byte_offset_tensor[25] = l_meta_data.m_tensor_offset_25;
        l_byte_offset_tensor[26] = l_meta_data.m_tensor_offset_26;
        l_byte_offset_tensor[27] = l_meta_data.m_tensor_offset_27;
        l_byte_offset_tensor[28] = l_meta_data.m_tensor_offset_28;
        l_byte_offset_tensor[29] = l_meta_data.m_tensor_offset_29;
        l_byte_offset_tensor[30] = l_meta_data.m_tensor_offset_30;
        l_byte_offset_tensor[31] = l_meta_data.m_tensor_offset_31;
    }

    GroupMemoryBarrierWithGroupSync();

    uint l_tensor_id = p_gid.x;
    uint l_tensor_out_offset = l_byte_offset_tensor[l_tensor_id+1];

    uint4 l_out_shape = l_tensors.Load4(l_tensor_out_offset + 4);
    uint l_start = 0;

    // to get the starting index for dim=1 for this particular thread
    for (uint l_i = 0; l_i < l_tensor_id; l_i++)
    {
        l_start += l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (2 + l_i));
    }
    // size of tensor for current thread
    uint l_ratio = l_attributes.Load(l_meta_data.m_attrib_offset + 4 * (2 + l_tensor_id));

    l_start *= 1024; // l_d_o x l_d_o
    l_start += 1 + 4; // rank + shape for input tensor

    uint l_n_elements = l_ratio * l_out_shape[2] * l_out_shape[3];
    uint l_inc = 4 * NN * GROUP_SIZE;
    
    for (l_i = 4 * (NN * p_gtid.x + p_gid.y); l_i < l_n_elements; l_i += l_inc)
    {
        uint4 l_temp = l_tensors.Load4(l_meta_data.m_tensor_offset_0 + 4 * (l_start + l_i) );
        l_tensors.Store4(l_tensor_out_offset + 4 * (5 + l_i), l_temp);
    }
}
