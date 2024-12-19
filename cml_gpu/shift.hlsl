// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

uint get_index(uint4 p_shape, uint p_ind_1, uint p_ind_2, uint p_ind_3)
{
    return (p_ind_3 + p_shape[3] * (p_ind_2 + p_shape[2] * p_ind_1));
}

int src_index(uint4 p_shape, uint p_ind_1, uint p_ind_2, uint p_ind_3)
{
    uint l_xs = p_shape[1] / 5;
    uint l_dir = p_ind_1 / l_xs; 
    int l_src_ind_2 = 0, l_src_ind_3 = 0;
    uint l_valid = 0;

    l_dir = (l_dir > 4) ? 4 : l_dir;

    switch (l_dir)
    {
    case 0:
        l_src_ind_2 = p_ind_2 - 1;
        l_src_ind_3 = p_ind_3;
        l_valid = (l_src_ind_2 >= 0);
        break;
    case 1:
        l_src_ind_2 = p_ind_2;
        l_src_ind_3 = p_ind_3 + 1;
        l_valid = (l_src_ind_3 < p_shape[3]);
        break;
    case 2:
        l_src_ind_2 = p_ind_2 + 1;
        l_src_ind_3 = p_ind_3;
        l_valid = (l_src_ind_2 < p_shape[2]);
        break;
    case 3:
        l_src_ind_2 = p_ind_2;
        l_src_ind_3 = p_ind_3 - 1;
        l_valid = (l_src_ind_3 >= 0);
        break;
    case 4:
        l_src_ind_2 = p_ind_2;
        l_src_ind_3 = p_ind_3;
        l_valid = 1;
        break;
    }

    return (l_valid == 1 ? ((int)(l_src_ind_3 + p_shape[3] * l_src_ind_2)) : (-99));
}

#define GROUP_SIZE 32
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_out_shape;
    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);

    uint l_ind_1 = p_gid.z;

    for(uint l_ind_2 = p_gtid.y; l_ind_2 < l_out_shape[2]; l_ind_2 += GROUP_SIZE)
    {
        for (uint l_ind_3 = p_gtid.x; l_ind_3 < l_out_shape[3]; l_ind_3 += GROUP_SIZE)
        {
            int l_src_index = src_index(l_in_shape_a, l_ind_1, l_ind_2, l_ind_3);
            float l_out = 0.0;

            if (l_src_index != -99)
            {
                l_out = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * ((uint)l_src_index + l_out_shape[2] * l_out_shape[3] * l_ind_1)));
            }
            l_tensors.Store(l_out_byte_offset + 4 * get_index(l_in_shape_a, l_ind_1, l_ind_2, l_ind_3), asuint(l_out));
        }
    }
}


//BACKUP
//groupshared float l_input[1024];
//
//#define GROUP_SIZE 32
//[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
//void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
//    uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
//{
//    // Make sure we are not in an error-state
//    CML_CHECK_KERNEL_ERROR;
//
//    // Every Tensor starts with its shape
//    uint4 l_in_shape_a;
//    uint l_in_byte_offset_a = l_meta_data.m_tensor_offset_0;
//    l_in_byte_offset_a += tensor_shape(l_tensors, l_in_byte_offset_a, l_in_shape_a);
//
//    uint4 l_out_shape;
//    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1;
//    l_out_byte_offset += tensor_shape(l_tensors, l_out_byte_offset, l_out_shape);
//
//    uint l_ind_3 = p_gtid.x;
//    uint l_ind_2 = p_gtid.y;
//    uint l_ind_1 = p_gid.z;
//
//    uint l_index = l_ind_3 + GROUP_SIZE * l_ind_2;  //0 to 1023
//
//    float l_temp = asfloat(l_tensors.Load(l_in_byte_offset_a + 4 * (l_index + l_out_shape[2] * l_out_shape[3] * p_gid.z)));
//    l_input[l_index] = l_temp;
//    GroupMemoryBarrierWithGroupSync();
//
//    if (l_ind_3 < l_out_shape[3] && l_ind_2 < l_out_shape[2])
//    {
//        int l_src_index = src_index2(l_in_shape_a, l_ind_1, l_ind_2, l_ind_3);
//        float l_out = 0.0;
//
//        if (l_src_index != -99)
//        {
//            l_out = input[(uint)l_src_index];
//        }
//        l_tensors.Store(l_out_byte_offset + 4 * get_index(l_in_shape_a, l_ind_1, l_ind_2, l_ind_3), asuint(l_out));
//    }
//}