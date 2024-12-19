// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 4
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Filter
// uint m_tensor_offset_2; // Bias
// uint m_tensor_offset_3; // Output

#include "conv_c_relu_func.hlsl"

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    conv_c_relu_func(p_gid, p_dtid, p_gtid, p_gi);
}
