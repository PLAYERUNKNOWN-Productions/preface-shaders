// Copyright:   PlayerUnknown Productions BV

#ifndef MBSHADER_CML_BINDINGS_H
#define MBSHADER_CML_BINDINGS_H

#include "../helper_shaders/mb_common.hlsl"
#include "cml_shared_buffers.hlsl"

#define CML_GET_BUFFERS     ConstantBuffer<cb_cml_meta_data_array_t> l_meta_data_cb = ResourceDescriptorHeap[g_push_constants.m_meta_data_cbv];\
                            cb_cml_meta_data_t l_meta_data = l_meta_data_cb.m_data[g_push_constants.m_meta_data_index];\
                            ByteAddressBuffer l_attributes = ResourceDescriptorHeap[g_push_constants.m_attribs_srv];\
                            RWByteAddressBuffer l_tensors = ResourceDescriptorHeap[g_push_constants.m_tensors_uav];\
                            RWByteAddressBuffer l_scratch = ResourceDescriptorHeap[g_push_constants.m_scratch_uav];

// Push constants
ConstantBuffer<cb_push_cml_t> g_push_constants : register(REGISTER_PUSH_CONSTANTS);

#endif // MBSHADER_CML_BINDINGS_H
