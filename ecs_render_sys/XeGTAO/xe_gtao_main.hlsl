///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation 
// 
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion", 
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// 
// Implementation:  Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>         (\_/)
// Version:         (see XeGTAO.h)                                                                            (='.'=)
// Details:         https://github.com/GameTechDev/XeGTAO                                                     (")_(")
//
// Version history: see XeGTAO.h
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../helper_shaders/mb_common.hlsl"
#include "xe_gtao_shared.hlsl"

ConstantBuffer<GTAOConstants> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#include "xe_gtao_imp.hlsl"

// input output textures for the first pass (XeGTAO_PrefilterDepths16x16)
//Texture2D<float>            g_srcRawDepth           : register( t0 );   // source depth buffer data (in NDC space in DirectX)
//RWTexture2D<lpfloat>        g_outWorkingDepthMIP0   : register( u0 );   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)
//RWTexture2D<lpfloat>        g_outWorkingDepthMIP1   : register( u1 );   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)
//RWTexture2D<lpfloat>        g_outWorkingDepthMIP2   : register( u2 );   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)
//RWTexture2D<lpfloat>        g_outWorkingDepthMIP3   : register( u3 );   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)
//RWTexture2D<lpfloat>        g_outWorkingDepthMIP4   : register( u4 );   // output viewspace depth MIP (these are views into g_srcWorkingDepth MIP levels)

// input output textures for the second pass (XeGTAO_MainPass)
//Texture2D<lpfloat>          g_srcWorkingDepth       : register( t0 );   // viewspace depth with MIPs, output by XeGTAO_PrefilterDepths16x16 and consumed by XeGTAO_MainPass
//Texture2D<uint>             g_srcNormalmap          : register( t1 );   // source normal map (if used)
//Texture2D<uint>             g_srcHilbertLUT         : register( t5 );   // hilbert lookup table  (if any)
//RWTexture2D<uint>           g_outWorkingAOTerm      : register( u0 );   // output AO term (includes bent normals if enabled - packed as R11G11B10 scaled by AO)
//RWTexture2D<unorm float>    g_outWorkingEdges       : register( u1 );   // output depth-based edges used by the denoiser
//RWTexture2D<uint>           g_outNormalmap          : register( u0 );   // output viewspace normals if generating from depth

// input output textures for the third pass (XeGTAO_Denoise)
//Texture2D<uint>             g_srcWorkingAOTerm      : register( t0 );   // coming from previous pass
//Texture2D<lpfloat>          g_srcWorkingEdges       : register( t1 );   // coming from previous pass
//RWTexture2D<uint>           g_outFinalAOTerm        : register( u0 );   // final AO term - just 'visibility' or 'visibility + bent normals'

// Engine-specific normal map loader
lpfloat3 LoadNormal( int2 pos )
{
    Texture2D<uint> srcNormalmap = ResourceDescriptorHeap[g_push_constants.SrcNormalmap];
#if 1
    // special decoding for external normals stored in 11_11_10 unorm - modify appropriately to support your own encoding 
    uint packedInput = srcNormalmap.Load( int3(pos, 0) ).x;
    float3 unpackedOutput = XeGTAO_R11G11B10_UNORM_to_FLOAT3( packedInput );
    float3 normal = normalize(unpackedOutput * 2.0.xxx - 1.0.xxx);
#else 
    // example of a different encoding
    float3 encodedNormal = srcNormalmap.Load(int3(pos, 0)).xyz;
    float3 normal = normalize(encodedNormal * 2.0.xxx - 1.0.xxx);
#endif

#if 0 // compute worldspace to viewspace here if your engine stores normals in worldspace; if generating normals from depth here, they're already in viewspace
    normal = mul( (float3x3)g_globals.View, normal );
#endif

    return (lpfloat3)normal;
}

// Engine-specific screen & temporal noise loader
lpfloat2 SpatioTemporalNoise( uint2 pixCoord, uint temporalIndex )    // without TAA, temporalIndex is always 0
{
    float2 noise;
#if 1   // Hilbert curve driving R2 (see https://www.shadertoy.com/view/3tB3z3)
    #ifdef XE_GTAO_HILBERT_LUT_AVAILABLE // load from lookup texture...
        uint index = g_srcHilbertLUT.Load( uint3( pixCoord % 64, 0 ) ).x;
    #else // ...or generate in-place?
        uint index = HilbertIndex( pixCoord.x, pixCoord.y );
    #endif
    index += 288*(temporalIndex%64); // why 288? tried out a few and that's the best so far (with XE_HILBERT_LEVEL 6U) - but there's probably better :)
    // R2 sequence - see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    return lpfloat2( frac( 0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114) ) );
#else   // Pseudo-random (fastest but looks bad - not a good choice)
    uint baseHash = Hash32( pixCoord.x + (pixCoord.y << 15) );
    baseHash = Hash32Combine( baseHash, temporalIndex );
    return lpfloat2( Hash32ToFloat( baseHash ), Hash32ToFloat( Hash32( baseHash ) ) );
#endif
}

// Engine-specific entry point for the first pass
[numthreads(8, 8, 1)]   // <- hard coded to 8x8; each thread computes 2x2 blocks so processing 16x16 block: Dispatch needs to be called with (width + 16-1) / 16, (height + 16-1) / 16
void CSPrefilterDepths16x16( uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupThreadID : SV_GroupThreadID )
{
    RWTexture2D<lpfloat> outWorkingDepthMIP0 = ResourceDescriptorHeap[g_push_constants.OutWorkingDepthMIP0];
    RWTexture2D<lpfloat> outWorkingDepthMIP1 = ResourceDescriptorHeap[g_push_constants.OutWorkingDepthMIP1];
    RWTexture2D<lpfloat> outWorkingDepthMIP2 = ResourceDescriptorHeap[g_push_constants.OutWorkingDepthMIP2];
    RWTexture2D<lpfloat> outWorkingDepthMIP3 = ResourceDescriptorHeap[g_push_constants.OutWorkingDepthMIP3];
    RWTexture2D<lpfloat> outWorkingDepthMIP4 = ResourceDescriptorHeap[g_push_constants.OutWorkingDepthMIP4];
    Texture2D<float> srcRawDepth = ResourceDescriptorHeap[g_push_constants.SrcRawDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];

    XeGTAO_PrefilterDepths16x16( dispatchThreadID, groupThreadID, g_push_constants, srcRawDepth, samplerPointClamp, outWorkingDepthMIP0, outWorkingDepthMIP1, outWorkingDepthMIP2, outWorkingDepthMIP3, outWorkingDepthMIP4 );
}

// Engine-specific entry point for the second pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSGTAOLow( const uint2 pixCoord : SV_DispatchThreadID )
{
    RWTexture2D<unorm float> outWorkingEdges = ResourceDescriptorHeap[g_push_constants.OutWorkingEdges];
    RWTexture2D<uint> outWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.OutWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingDepth = ResourceDescriptorHeap[g_push_constants.SrcWorkingDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];

    XeGTAO_MainPass( pixCoord, 1, 2, SpatioTemporalNoise(pixCoord, g_push_constants.NoiseIndex), LoadNormal(pixCoord), g_push_constants, srcWorkingDepth, samplerPointClamp, outWorkingAOTerm, outWorkingEdges );
}

// Engine-specific entry point for the second pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSGTAOMedium( const uint2 pixCoord : SV_DispatchThreadID )
{
    RWTexture2D<unorm float> outWorkingEdges = ResourceDescriptorHeap[g_push_constants.OutWorkingEdges];
    RWTexture2D<uint> outWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.OutWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingDepth = ResourceDescriptorHeap[g_push_constants.SrcWorkingDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];

    XeGTAO_MainPass( pixCoord, 2, 2, SpatioTemporalNoise(pixCoord, g_push_constants.NoiseIndex), LoadNormal(pixCoord), g_push_constants, srcWorkingDepth, samplerPointClamp, outWorkingAOTerm, outWorkingEdges );
}

// Engine-specific entry point for the second pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSGTAOHigh( const uint2 pixCoord : SV_DispatchThreadID )
{
    RWTexture2D<unorm float> outWorkingEdges = ResourceDescriptorHeap[g_push_constants.OutWorkingEdges];
    RWTexture2D<uint> outWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.OutWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingDepth = ResourceDescriptorHeap[g_push_constants.SrcWorkingDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];

    XeGTAO_MainPass( pixCoord, 3, 3, SpatioTemporalNoise(pixCoord, g_push_constants.NoiseIndex), LoadNormal(pixCoord), g_push_constants, srcWorkingDepth, samplerPointClamp, outWorkingAOTerm, outWorkingEdges );
}

// Engine-specific entry point for the second pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSGTAOUltra( const uint2 pixCoord : SV_DispatchThreadID )
{
    RWTexture2D<unorm float> outWorkingEdges = ResourceDescriptorHeap[g_push_constants.OutWorkingEdges];
    RWTexture2D<uint> outWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.OutWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingDepth = ResourceDescriptorHeap[g_push_constants.SrcWorkingDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];

    XeGTAO_MainPass( pixCoord, 9, 3, SpatioTemporalNoise( pixCoord, g_push_constants.NoiseIndex ), LoadNormal( pixCoord ), g_push_constants, srcWorkingDepth, samplerPointClamp, outWorkingAOTerm, outWorkingEdges );
}

// Engine-specific entry point for the third pass
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSDenoisePass( const uint2 dispatchThreadID : SV_DispatchThreadID )
{
    Texture2D<uint> srcWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.SrcWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingEdges = ResourceDescriptorHeap[g_push_constants.SrcWorkingEdges];
    RWTexture2D<uint> outFinalAOTerm = ResourceDescriptorHeap[g_push_constants.OutFinalAOTerm];

    const uint2 pixCoordBase = dispatchThreadID * uint2( 2, 1 );    // we're computing 2 horizontal pixels at a time (performance optimization)

    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];
    XeGTAO_Denoise( pixCoordBase, g_push_constants, srcWorkingAOTerm, srcWorkingEdges, samplerPointClamp, outFinalAOTerm, false );
}

[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSDenoiseLastPass( const uint2 dispatchThreadID : SV_DispatchThreadID )
{
    Texture2D<uint> srcWorkingAOTerm = ResourceDescriptorHeap[g_push_constants.SrcWorkingAOTerm];
    Texture2D<lpfloat> srcWorkingEdges = ResourceDescriptorHeap[g_push_constants.SrcWorkingEdges];
    RWTexture2D<uint> outFinalAOTerm = ResourceDescriptorHeap[g_push_constants.OutFinalAOTerm];
    
    const uint2 pixCoordBase = dispatchThreadID * uint2( 2, 1 );    // we're computing 2 horizontal pixels at a time (performance optimization)

    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];
    XeGTAO_Denoise( pixCoordBase, g_push_constants, srcWorkingAOTerm, srcWorkingEdges, samplerPointClamp, outFinalAOTerm, true );
}

// Optional screen space viewspace normals from depth generation
[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void CSGenerateNormals( const uint2 pixCoord : SV_DispatchThreadID )
{
    Texture2D<float> srcRawDepth = ResourceDescriptorHeap[g_push_constants.SrcRawDepth];
    SamplerState samplerPointClamp = SamplerDescriptorHeap[SAMPLER_LINEAR_CLAMP];
    
    float3 viewspaceNormal = XeGTAO_ComputeViewspaceNormal( pixCoord, g_push_constants, srcRawDepth, samplerPointClamp );

    // pack from [-1, 1] to [0, 1] and then to R11G11B10_UNORM
    RWTexture2D<uint> outNormalmap = ResourceDescriptorHeap[g_push_constants.OutNormalmap];
    outNormalmap[ pixCoord ] = XeGTAO_FLOAT3_to_R11G11B10_UNORM( saturate( viewspaceNormal * 0.5 + 0.5 ) );
}
///
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
