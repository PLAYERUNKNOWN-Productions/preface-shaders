// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_GRASS_COMMON_H
#define MB_SHADER_GRASS_COMMON_H

static const uint GRASS_TYPE_ID_INVALID = 0;
static const uint GRASS_TYPE_ID_DEFAULT = 1;

static const uint VERTICES_PER_BLADE_EDGE = 4;
static const uint VERTICES_PER_BLADE =  2 * VERTICES_PER_BLADE_EDGE;
static const uint TRIANGLES_PER_BLADE = 6;
static const uint MAX_BLADE_COUNT = 32;

static const float RANDOM_EXTRA_HEIGHT_SCALE = 0.1f;
static const float RANDOM_EXTRA_PATCH_OFFSET = 1.0f; // We added a random offset of range [0, 1] in mb_grass_patch_preparation.hlsl

// All grass beyond this distance will use the lowest LOD. Note that they still exists.
static const float GRASS_LOD_END_DISTANCE = 20.0f;

static const uint GRASS_LOD_LEVEL_1_BLADE_COUNT = 2;

#endif // MB_SHADER_GRASS_COMMON_H
