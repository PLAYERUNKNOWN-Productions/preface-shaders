// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

#define SMAA_PIXEL_SIZE float2(1.0 / g_push_constants.m_render_target_width, 1.0 / g_push_constants.m_render_target_height) //todo make this tweakable?
#define SMAA_HLSL_4_1 1
#define SMAA_PRESET_ULTRA 1

#define SMAA_PREDICATION 1