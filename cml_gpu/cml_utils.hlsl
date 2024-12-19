// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MBSHADER_CML_UTILS_H
#define MBSHADER_CML_UTILS_H

#include "cml_shader_buffers.hlsl"

#define FLOAT_SIZE 4
#define INT_SIZE 4
#define MB_CML_GPU_MAX_TENSOR_RANK 8

//-----------------------------------------------------------------------------
// Group size related defines
#ifndef GROUP_SIZE_UNARY
    #define GROUP_SIZE_UNARY 1024
#endif
#define DEF_THREAD_GROUP_SIZE_UNARY [numthreads(GROUP_SIZE_UNARY, 1, 1)]

#ifndef GROUP_SIZE_BINARY_X
    #define GROUP_SIZE_BINARY_X 16
    #define GROUP_SIZE_BINARY_Y 16
#endif
#define DEF_THREAD_GROUP_SIZE_BINARY [numthreads(GROUP_SIZE_BINARY_X, GROUP_SIZE_BINARY_Y, 1)]

#ifndef GROUP_SIZE_AT_X
    #define GROUP_SIZE_AT_X 16
    #define GROUP_SIZE_AT_Y 16
#endif
#define DEF_THREAD_GROUP_SIZE_AT [numthreads(GROUP_SIZE_AT_X, GROUP_SIZE_AT_Y, 1)]

//-----------------------------------------------------------------------------
uint float_attrib(in ByteAddressBuffer p_attrib_buffer, in uint p_value_offset, out float p_attrib)
{
    p_attrib = asfloat(p_attrib_buffer.Load(p_value_offset));
    return 4;
}

//-----------------------------------------------------------------------------
uint int_attrib(in ByteAddressBuffer p_attrib_buffer, in uint p_value_offset, out int p_attrib)
{
    p_attrib = asint(p_attrib_buffer.Load(p_value_offset));
    return 4;
}

//-----------------------------------------------------------------------------
uint attrib_count(in ByteAddressBuffer p_attrib_buffer, in uint p_base_offset)
{
    return p_attrib_buffer.Load(p_base_offset);
}

//-----------------------------------------------------------------------------
bool shape_equal(in uint4 p_shape_a, in uint4 p_shape_b)
{
    for (uint l_index = 0; l_index < 4; ++l_index)
    {
        if (p_shape_a[l_index] != p_shape_b[l_index])
        {
            return false;
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
uint thread_id_to_byte_index(in uint3 p_thread_id, in uint4 p_shape)
{
    return (4 * (min(p_thread_id.x, p_shape[1] - 1) * p_shape[3] * p_shape[2] +
                 min(p_thread_id.y, p_shape[2] - 1) * p_shape[3] +
                 min(p_thread_id.z, p_shape[3] - 1)));
}

//-----------------------------------------------------------------------------
// Read the shape from the buffer at p_offset. Returns the number of bytes read.
uint tensor_shape_prepend(in RWByteAddressBuffer p_buffer, in uint p_offset, out uint4 p_shape)
{
    uint l_offset = p_offset;
    p_shape = uint4(1, 1, 1, 1);

    uint l_rank = p_buffer.Load(l_offset);
    l_offset += 4;

    if ((l_rank > 4) || (l_rank < 1))
    {
        p_shape = uint4(0, 0, 0, 0);
    }
    else
    {
        if (l_rank > 3)
        {
            p_shape[0] = p_buffer.Load(l_offset);
            l_offset += 4;
        }

        if (l_rank > 2)
        {
            p_shape[1] = p_buffer.Load(l_offset);
            l_offset += 4;
        }

        if (l_rank > 1)
        {
            p_shape[2] = p_buffer.Load(l_offset);
            l_offset += 4;
        }

        p_shape[3] = p_buffer.Load(l_offset);
        l_offset += 4;
    }

    return (l_offset - p_offset);
}

//-----------------------------------------------------------------------------
// Read the shape from the buffer at p_offset. Returns the number of bytes read.
uint tensor_shape(in RWByteAddressBuffer p_buffer, in uint p_offset, out uint4 p_shape)
{
    uint l_offset = p_offset;
    p_shape = uint4(1, 1, 1, 1);

    uint l_rank = p_buffer.Load(l_offset);
    l_offset += 4;

    if ((l_rank > 4) || (l_rank < 1))
    {
        p_shape = uint4(0, 0, 0, 0);
    }
    else
    {
        p_shape[0] = p_buffer.Load(l_offset);
        l_offset += 4;

        if (l_rank > 1)
        {
            p_shape[1] = p_buffer.Load(l_offset);
            l_offset += 4;
        }

        if (l_rank > 2)
        {
            p_shape[2] = p_buffer.Load(l_offset);
            l_offset += 4;
        }

        if (l_rank > 3)
        {
            p_shape[3] = p_buffer.Load(l_offset);
            l_offset += 4;
        }
    }

    return (l_offset - p_offset);
}

//-----------------------------------------------------------------------------
uint tensor_shape(in RWByteAddressBuffer p_buffer, in uint p_offset, out uint2 p_shape)
{
    uint l_offset = p_offset;
    p_shape = uint2(1, 1);

    uint l_rank = p_buffer.Load(l_offset);
    l_offset += 4;

    if ((l_rank > 2) || (l_rank < 1))
    {
        p_shape = uint2(0, 0);
    }
    else
    {
        p_shape[0] = p_buffer.Load(l_offset);
        l_offset += 4;

        if (l_rank > 1)
        {
            p_shape[1] = p_buffer.Load(l_offset);
            l_offset += 4;
        }
    }

    return (l_offset - p_offset);
}

//-----------------------------------------------------------------------------
uint tensor_shape(in RWByteAddressBuffer p_buffer, in uint p_offset, out uint p_shape[6])
{
    uint4 l_val_03 = uint4(1, 1, 1, 1);
    uint2 l_val_45 = uint2(1, 1);

    uint l_rank = asuint(p_buffer.Load(p_offset));
    p_offset += 4;

    if (l_rank == 6)
    {
        l_val_03 = p_buffer.Load4(p_offset);
        p_offset += 16;
        l_val_45 = p_buffer.Load2(p_offset);
        p_offset += 8;
    }
    else if (l_rank == 5)
    {
        l_val_03 = p_buffer.Load4(p_offset);
        p_offset += 16;
        l_val_45.x = p_buffer.Load(p_offset);
        p_offset += 4;
    }
    else if (l_rank == 4)
    {
        l_val_03 = p_buffer.Load4(p_offset);
        p_offset += 16;
    }
    else if (l_rank == 3)
    {
        l_val_03.xyz = p_buffer.Load3(p_offset);
        p_offset += 12;
    }
    else if (l_rank == 2)
    {
        l_val_03.xy = p_buffer.Load2(p_offset);
        p_offset += 8;
    }
    else if (l_rank == 1)
    {
        l_val_03.x = p_buffer.Load(p_offset);
        p_offset += 4;
    }
    else
    {
        l_val_03 = uint4(0, 0, 0, 0);
        l_val_45 = uint2(0, 0);
    }

    p_shape[0] = l_val_03[0];
    p_shape[1] = l_val_03[1];
    p_shape[2] = l_val_03[2];
    p_shape[3] = l_val_03[3];
    p_shape[4] = l_val_45[0];
    p_shape[5] = l_val_45[1];

    return p_offset;
}

//-----------------------------------------------------------------------------
// Calculate the number of items in a tensor with p_shape
uint shape_size(in uint4 p_shape)
{
	return (p_shape[0] * p_shape[1] * p_shape[2] * p_shape[3]);
}

uint shape_size(in uint p_shape[MB_CML_GPU_MAX_TENSOR_RANK], in uint p_rank)
{
    uint l_shape_size = 1;
    for(uint l_i = 0; l_i < p_rank; l_i++)
        l_shape_size = p_shape[l_i];

    return l_shape_size;
}

//-----------------------------------------------------------------------------
// Calculate the number of bytes in a tensor with p_shape
uint tensor_size(in uint4 p_shape)
{
    return (4 * shape_size(p_shape));
}

//-----------------------------------------------------------------------------
bool prepare_unary_operation(in cb_cml_meta_data_t p_meta_data, in RWByteAddressBuffer p_tensors,
                             in uint3 p_dispatch_thread_id, out uint p_out_byte_offset, out float p_a_in)
{
    // Every Tensor starts with its shape
    uint4 l_in_shape_a;
    uint l_in_byte_offset_a = p_meta_data.m_tensor_offset_0;
    l_in_byte_offset_a += tensor_shape(p_tensors, l_in_byte_offset_a, l_in_shape_a);

    uint4 l_out_shape;
    p_out_byte_offset = p_meta_data.m_tensor_offset_1;
    p_out_byte_offset += tensor_shape(p_tensors, p_out_byte_offset, l_out_shape);

    if (p_dispatch_thread_id.x >= l_out_shape[1] * l_out_shape[2] * l_out_shape[3])
    {
        return false;
    }

    // Calculate local index for the current thread
    uint l_byte_index = 4 * p_dispatch_thread_id.x;

    // From local to global index for the tensor
    l_in_byte_offset_a += l_byte_index; // Same shape as output
    p_out_byte_offset += l_byte_index;

    // Get operants
    p_a_in = asfloat(p_tensors.Load(l_in_byte_offset_a));

    return true;
}

// Convert flattened output coordinate to tensor coordinate
void out_flatten_to_coord(in uint p_out_flat, in uint p_out_shape[MB_CML_GPU_MAX_TENSOR_RANK],
                          in uint p_rank, out uint p_out_coord[MB_CML_GPU_MAX_TENSOR_RANK])
{
    uint l_temp1 = p_out_flat;

    // Find unflatten index (coordinate) for input
    for (int l_i = p_rank - 1; l_i >= 0; l_i--)
    {
        p_out_coord[l_i] = l_temp1 % p_out_shape[l_i];
        l_temp1 /= p_out_shape[l_i];
    }

    for (l_i = MB_CML_GPU_MAX_TENSOR_RANK - 1; l_i >= p_rank; l_i--)
    {
        p_out_coord[l_i] = 0;
    }
}

// Convert flattened output coordinate to tensor coordinate
int in_coord_to_flatten_unary(in uint p_z_coord[MB_CML_GPU_MAX_TENSOR_RANK],
                              in uint p_in_shape[MB_CML_GPU_MAX_TENSOR_RANK],
                              in uint p_rank,
                              in bool p_broadcast[MB_CML_GPU_MAX_TENSOR_RANK])
{
    int l_flat_idx = 0;

    if (!p_broadcast[0])
        l_flat_idx += p_z_coord[0];

    for (uint l_i = 1; l_i < p_rank; l_i++)
    {
        l_flat_idx = l_flat_idx * p_in_shape[l_i];
        if (!p_broadcast[l_i])
            l_flat_idx += p_z_coord[l_i];
    }

    return l_flat_idx;
}

// Convert input tensor coordinate to flatten
int in_coord_to_flatten(in uint in_coord[MB_CML_GPU_MAX_TENSOR_RANK], in uint in_shape[MB_CML_GPU_MAX_TENSOR_RANK], in uint rank)
{
    int flat_idx = in_coord[0];

    for (uint i = 1; i < rank; i++)
    {
        flat_idx = flat_idx * in_shape[i] + in_coord[i];
    }

    return flat_idx;
}

// Get shape of tensor
void get_tensor_shape(in RWByteAddressBuffer p_tensors, in uint p_byte_offset_tensor,
    in uint p_rank, out uint p_shape[MB_CML_GPU_MAX_TENSOR_RANK])
{
    for (uint l_j = 0; l_j < p_rank; l_j++)
    {
        p_shape[l_j] = asuint(p_tensors.Load(p_byte_offset_tensor + FLOAT_SIZE * (1 + l_j)));
    }
    // fill remaining shape as 1 (i.e. unsqueeze)
    for (l_j = p_rank; l_j < MB_CML_GPU_MAX_TENSOR_RANK; l_j++)
    {
        p_shape[l_j] = 1;
    }
}

// Get total number of elements in a tensor
uint get_shape_size(in uint p_shape[MB_CML_GPU_MAX_TENSOR_RANK], in uint p_rank)
{
    uint l_n_outputs = 1;

    for (uint l_i = 0; l_i < p_rank; l_i++)
    {
        l_n_outputs *= p_shape[l_i];
    }
    return l_n_outputs;
}

#endif // MBSHADER_CML_UTILS_H
