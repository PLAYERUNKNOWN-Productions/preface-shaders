// Copyright:   PlayerUnknown Productions BV

#include "cml_bindings.hlsl"
#include "cml_utils.hlsl"
#include "cml_error.hlsl"

// uint m_tensor_count;    // 2
// uint m_tensor_offset_0; // Input
// uint m_tensor_offset_1; // Output

#define l_in_ch (l_in_shape_input[1])
#define l_d_f (l_attribs.m_size.z)
#define l_s1 (l_attribs.m_stride.x)
#define l_s2 (l_attribs.m_stride.y)
#define l_d1 (l_attribs.m_dilation.x)
#define l_d2 (l_attribs.m_dilation.y)
#define l_p1 (l_attribs.m_padding.x)
#define l_p2 (l_attribs.m_padding.z)

#define GROUP_SIZE 16
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void cs_main(uint3 p_gid : SV_GroupID, uint3 p_dtid : SV_DispatchThreadID,
             uint3 p_gtid : SV_GroupThreadID, uint p_gi : SV_GroupIndex)
{
    CML_GET_BUFFERS;

    // Make sure we are not in an error-state
    CML_CHECK_KERNEL_ERROR;

    // Get the attributes
    uint4 l_attrib_size = asuint(l_attributes.Load4(l_meta_data.m_attrib_offset + 4));
    float l_attrib_alpha = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 20));
    float l_attrib_beta = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 24));
    float l_attrib_bias = asfloat(l_attributes.Load(l_meta_data.m_attrib_offset + 28));

    // Every Tensor starts with its shape
    uint4 l_in_shape_input;
    uint l_in_byte_offset_input = l_meta_data.m_tensor_offset_0;
    l_in_byte_offset_input += tensor_shape(l_tensors, l_in_byte_offset_input, l_in_shape_input);

    uint l_out_byte_offset = l_meta_data.m_tensor_offset_1 + 20;

    uint l_k2 = p_gid.z;
    uint l_i1 = p_gid.y * GROUP_SIZE + p_gtid.y;
    uint l_i2 = p_gid.x * GROUP_SIZE + p_gtid.x;

    int l_d_i = (int)(l_in_shape_input[3]);
    int l_half_size = l_attrib_size.y / 2;

    if (l_k2 < l_in_ch && l_i2 < l_in_shape_input[3] && l_i1 < l_in_shape_input[2])
    {
        float l_avg = 0.0f;

        for (int l_j2 = 0; l_j2 < l_attrib_size.y; l_j2++)
        {
            int l_ch = l_k2 - l_half_size + l_j2;
            int l_idx_input = l_i2 + l_i1 * l_d_i + l_ch * l_d_i * l_d_i;

            if (l_ch >= 0 && l_ch < l_in_ch)
            {
                float l_t1 = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * l_idx_input));
                l_avg += l_t1 * l_t1;
            }
        }

        int l_idx_output = l_i2 + l_i1 * l_d_i + l_k2 * l_d_i * l_d_i;
        float l_value = asfloat(l_tensors.Load(l_in_byte_offset_input + 4 * l_idx_output));

        l_value /= pow(l_avg * l_attrib_alpha / l_attrib_size.y + l_attrib_bias, l_attrib_beta);

        l_tensors.Store(l_out_byte_offset + 4 * l_idx_output, asuint(l_value));
    }
}
