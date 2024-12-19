// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#ifndef MBSHADER_CML_SHADER_BUFFERS_H
#define MBSHADER_CML_SHADER_BUFFERS_H

#include "../shared_shaders/mb_shared_common.hlsl"

// This is HLSL shared with C implementation
// Use of only shared functionality is permited!

//-----------------------------------------------------------------------------
// Define math types
//-----------------------------------------------------------------------------

#ifdef __cplusplus

// Define types to match with C++ code
#define float3x3 mb_math::float3x3
#define float4x4 mb_math::float4x4
#define float4x3 mb_math::float4x3
#define float2 mb_math::float2
#define float3 mb_math::float3
#define float4 mb_math::float4
#define int2 mb_math::int2
#define int3 mb_math::int3
#define int4 mb_math::int4
#define uint2 mb_math::uint2
#define uint3 mb_math::uint3
#define uint4 mb_math::uint4
#define uint uint32_t
#define int int32_t

// Define macro for passing structures as push constants
#define RAL_PUSH_CONSTANTS(l_push_constants) sizeof(l_push_constants), &l_push_constants

// Constant buffer rules are strict and cannot be fully emulated on C++ side
// This results in CPU/GPU structure alignment to be different and unpredictable bugs
// The closes way to emulate is is to use pragma pack(4)
// For more details refer to https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
//! \todo DXC compiler currently does not support turning off cb packing, but it might in the future
#pragma pack(push, 4)

namespace mb_shader_buffers
{
#endif

//-----------------------------------------------------------------------------
// Constant buffers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
struct cb_cml_meta_data_t
{
    uint m_attrib_count;    // Number of attributes
    uint m_attrib_offset;   // Offset into l_attributes buffer to start of tightly packed attributes
    uint m_tensor_count;    // Number of input/output tensors
    uint m_tensor_offset_0; // Tensors
    uint m_tensor_offset_1;
    uint m_tensor_offset_2;
    uint m_tensor_offset_3;
    uint m_tensor_offset_4;
    uint m_tensor_offset_5;
    uint m_tensor_offset_6;
    uint m_tensor_offset_7;
    uint m_tensor_offset_8;
    uint m_tensor_offset_9;
    uint m_tensor_offset_10;
    uint m_tensor_offset_11;
    uint m_tensor_offset_12;
    uint m_tensor_offset_13;
    uint m_tensor_offset_14;
    uint m_tensor_offset_15;
    uint m_tensor_offset_16; // Tensors
    uint m_tensor_offset_17;
    uint m_tensor_offset_18;
    uint m_tensor_offset_19;
    uint m_tensor_offset_20;
    uint m_tensor_offset_21;
    uint m_tensor_offset_22;
    uint m_tensor_offset_23;
    uint m_tensor_offset_24;
    uint m_tensor_offset_25;
    uint m_tensor_offset_26;
    uint m_tensor_offset_27;
    uint m_tensor_offset_28;
    uint m_tensor_offset_29;
    uint m_tensor_offset_30;
    uint m_tensor_offset_31;

    uint m_padding_0;
    uint4 m_padding_1;
    uint4 m_padding_2;
    uint4 m_padding_3;
    uint4 m_padding_4;
    uint4 m_padding_5;
    uint4 m_padding_6;
    uint4 m_padding_7;
};

struct cb_cml_meta_data_array_t
{
    cb_cml_meta_data_t m_data[256]; // Maximum CBV size is 65536. So maximum number is 65536 / 256 = 256.
};

//-----------------------------------------------------------------------------
// Push constants
struct cb_push_cml_t
{
    uint m_meta_data_cbv;
    uint m_meta_data_index;
    uint m_tensors_uav;
    uint m_scratch_uav;
    uint m_attribs_srv;
    uint m_error_uav;
};

//-----------------------------------------------------------------------------
// Undefine math types

#ifdef __cplusplus
};

// Pop packing rules(see upper side of this file)
#pragma pack(pop)

// Undefine types to keep math types in a separate namespace
#undef float3x3
#undef float4x4
#undef float4x3
#undef float2
#undef float3
#undef float4
#undef int2
#undef int3
#undef int4
#undef uint2
#undef uint3
#undef uint4
#undef uint
#undef int
#endif

#endif // MBSHADER_CML_SHADER_BUFFERS_H