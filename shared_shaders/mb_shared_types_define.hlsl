// Copyright:   PlayerUnknown Productions BV

// Define types to match with C++ code
//#define float3x3 mb_math::float3x3 - Due to alignment, a float3x3 in a constant buffer is a float4x3: as this is error prone, not allowed for now
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
