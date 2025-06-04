// Copyright:   PlayerUnknown Productions BV

//todo make this tweakable?
#define SMAA_RT_METRICS float4(1.f / g_push_constants.m_render_target_width,  \
                               1.f / g_push_constants.m_render_target_height, \
                                     g_push_constants.m_render_target_width,  \
                                     g_push_constants.m_render_target_height)
#define SMAA_HLSL_4_1 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_PREDICATION 1
