#include "../helper_shaders/mb_common.hlsl"

#ifndef MB_TEXTURE_TYPE
#   define MB_TEXTURE_TYPE float
#endif

#ifndef MB_NUM_CHANNELS
#   define MB_NUM_CHANNELS 4
#endif

#if MB_NUM_CHANNELS == 1
#   define MB_SWIZZLE x    
#elif MB_NUM_CHANNELS == 2
#   define MB_SWIZZLE xy
#elif MB_NUM_CHANNELS == 3
#   define MB_SWIZZLE xyz
#elif MB_NUM_CHANNELS == 4
#   define MB_SWIZZLE xyzw
#else
#   pragma error unexpected number of channels
#endif

//-----------------------------------------------------------------------------
// Resources
//-----------------------------------------------------------------------------

ConstantBuffer<cb_push_memset_uav> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

//-----------------------------------------------------------------------------
// CS
//-----------------------------------------------------------------------------

[numthreads(MB_MEMSET_UAV_THREADGROUP_SIZE_X, MB_MEMSET_UAV_THREADGROUP_SIZE_Y, 1)]
void cs_main(uint3 p_dispatch_thread_id : SV_DispatchThreadID)
{
    RWTexture2D<CONCAT(MB_TEXTURE_TYPE, MB_NUM_CHANNELS)> l_texture = ResourceDescriptorHeap[g_push_constants.m_texture_uav];
    l_texture[p_dispatch_thread_id.xy] = g_push_constants.m_value.MB_SWIZZLE;
}
