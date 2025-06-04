// Copyright:   PlayerUnknown Productions BV

#ifndef MB_SHADER_SHARED_COMMON
#define MB_SHADER_SHARED_COMMON

// This is HLSL shared with C implementation
// Use of only shared functionality is permited!

// Reserved descriptors in the heap
// IMPORTANT: Indices MUST NOT overlap. It can lead to a GPU crash

// Samplers
#define SAMPLER_POINT_CLAMP                     0
#define SAMPLER_POINT_WRAP                      1
#define SAMPLER_POINT_MIRROR                    2
#define SAMPLER_LINEAR_CLAMP                    3
#define SAMPLER_LINEAR_WRAP                     4
#define SAMPLER_LINEAR_MIRROR                   5
#define SAMPLER_ANISO16_CLAMP                   6
#define SAMPLER_ANISO16_WRAP                    7
#define SAMPLER_ANISO16_MIRROR                  8
#define SAMPLER_COMPARISON_LINEAR_LESS          9
#define SAMPLER_COMPARISON_LINEAR_GREATER       10
#define SAMPLER_MIN_CLAMP                       11

// CBV/SRV/UAV
#define CBV_TEST_VALUES                         12
#define DEBUG_RENDERING_TRIANGLES_UAV           (CBV_TEST_VALUES + 1)
#define DEBUG_RENDERING_TRIANGLES_COUNTER_UAV   (DEBUG_RENDERING_TRIANGLES_UAV + 1)
#define DEBUG_RENDERING_LINES_UAV               (DEBUG_RENDERING_TRIANGLES_COUNTER_UAV + 1)
#define DEBUG_RENDERING_LINES_COUNTER_UAV       (DEBUG_RENDERING_LINES_UAV + 1)
#define RAYTRACING_FEEDBACK_BUFFER_COUNTER_UAV  (DEBUG_RENDERING_LINES_COUNTER_UAV + 1)
#define RAYTRACING_FEEDBACK_BUFFER_COUNTER_SRV  (RAYTRACING_FEEDBACK_BUFFER_COUNTER_UAV + 1)
#define RAYTRACING_FEEDBACK_BUFFER_UAV          (RAYTRACING_FEEDBACK_BUFFER_COUNTER_SRV + 1)
#define RAYTRACING_FEEDBACK_BUFFER_SRV          (RAYTRACING_FEEDBACK_BUFFER_UAV + 1)

// Push constant space
#define MB_PUSH_CONSTANTS_ROOT_SLOT 0
#define MB_PUSH_CONSTANTS_COUNT 64
#define MB_PUSH_CONSTANTS_SPACE 999

#define MB_MAX_DIRECTIONAL_LIGHTS (2)
#define MB_MAX_PUNCTUAL_LIGHTS (200)
#define MB_MAX_LIGHTS (MB_MAX_DIRECTIONAL_LIGHTS + MB_MAX_PUNCTUAL_LIGHTS)

#define LIGHT_TYPE_DIRECTIONAL (0)
#define LIGHT_TYPE_OMNI        (1)
#define LIGHT_TYPE_SPOT        (2)
#define LIGHT_TYPE_NULL        (0xFFFFFFFF)

// Postprocessing
#define MB_MAX_ADAPTED_LUM_FRAMES (256)
#define MB_HDR_USE_FLOAT32 (0)

#define POSTPROCESS_GENERAL_THREAD_GROUP_SIZE 16
#define COPY_THREAD_GROUP_SIZE 16
#define DOWNSAMPLE_THREAD_GROUP_SIZE 16
#define SPOT_METER_THREAD_GROUP_SIZE 16

#define TONEMAP_OP_EXP               0
#define TONEMAP_OP_REINHARD          1
#define TONEMAP_OP_REINHARD_JODIE    2
#define TONEMAP_OP_UNCHARTED2_FILMIC 3
#define TONEMAP_OP_ACES_APPROX       4
#define TONEMAP_OP_ACES_FITTED       5

#define MB_EXPOSURE_FUSION_USE_GUIDED_FILTERING (0)

// Quadtree
#define TILE_GENERATION_THREADGROUP_SIZE 16
#define VT_GENERATION_THREADGROUP_SIZE 16
#define TILE_NO_PARENT 0xFFFFFFFF
#define MAX_LAYER_COUNT 16
#define TILE_POPULATION_THREADGROUP_SIZE 64
#define TILE_POPULATION_UPDATE_THREADGROUP_SIZE 32
#define TILE_MESH_BAKING_THREAD_GROUP_SIZE 32

#define PROCEDURAL_SPLAT_THREADGROUP_SIZE 16
#define PROCEDURAL_POPULATION_THREADGROUP_SIZE 16

#define GENERATE_DFG_THREAD_GROUP_SIZE 16

#define MB_HI_Z_THREADGROUP_SIZE 32

#define MB_GSM_THREADGROUP_SIZE_X 8
#define MB_GSM_THREADGROUP_SIZE_Y 4

#define MB_MEMSET_UAV_THREADGROUP_SIZE_X 8
#define MB_MEMSET_UAV_THREADGROUP_SIZE_Y 8

#define INSTANCING_NO_USER_DATA -1
#define INSTANCING_THREADGROUP_SIZE 16

#define ATMOSPHERIC_SCATTERING_THREAD_GROUP_SIZE 8
#define GENERATE_SCATTERING_LUT_THREAD_GROUP_SIZE 16

#define COMPRESS_THREAD_GROUP_SIZE 8

#define RAYTRACING_DENOISING_THREAD_GROUP_SIZE 8
#define RAYTRACING_VT_FEEDBACK_THREADGROUP_SIZE 16
#define RAYTRACING_UPSAMPLING_THREAD_GROUP_SIZE 8

#define RECONSTRUCT_NORMAL_THREAD_GROUP_SIZE 32
#define SSAO_THREAD_GROUP_SIZE 32
#define GTAO_THREAD_GROUP_SIZE 32

#define LIGHTING_COMBINATION_THREAD_GROUP_SIZE 16

#define DEFERRED_LIGHTING_THREAD_GROUP_SIZE_X 8
#define DEFERRED_LIGHTING_THREAD_GROUP_SIZE_Y 8

enum lighting_classification_types_t
{
    e_lighting_classification_types_default,
    e_lighting_classification_types_terrain,
    e_lighting_classification_types_translucent_lighting,
    e_lighting_classification_types_count,
};

enum lighting_classification_mask_t
{
    e_lighting_classification_mask_default              = 0x1u << e_lighting_classification_types_default,
    e_lighting_classification_mask_terrain              = 0x1u << e_lighting_classification_types_terrain,
    e_lighting_classification_mask_translucent_lighting = 0x1u << e_lighting_classification_types_translucent_lighting,
};

static lighting_classification_mask_t get_classification_mask_from_type(const lighting_classification_types_t p_classification_type)
{
    return (lighting_classification_mask_t)(0x1u << (uint32_t)p_classification_type);
}

#define MOTION_BLUR_THREAD_GROUP_SIZE 8

#define IMPOSTOR_THREADGROUP_SIZE 32

#define MB_GRASS_PATCH_THREADGROUP_SIZE 32
#define MB_GRASS_THREADGROUP_SIZE 32

// Population processing
#define POPULATION_COMPACT_COPY_THREADGROUP_SIZE 32
#define POPULATION_COMPACT_RESET_THREADGROUP_SIZE 32
#define POPULATION_COMPACT_APPEND_THREADGROUP_SIZE 32

// We cannot clear selection pass with 0xFFFFFFFF
// Instead we shift entity IDs by 1
inline uint32_t pack_entity_id(uint32_t enity_id)
{
    return enity_id + 1;
}
inline uint32_t unpack_entity_id(uint32_t enity_id)
{
    return enity_id - 1;
}

#endif
