// Copyright:   PlayerUnknown Productions BV

// See mb_cml_gpu_utils.cpp, line 196, init_inference_gpu()
// Create a buffer for error, debug en develpoment feedback
// Type   | byte-offset | byte-count  | word-count | description
// uint     0             4             1            error-count
// int      4             28            7            int numbers
// float    32            32            8            float numbers
// char     64            959           -            text characters, null terminated ('\0')
// char     1023          1             -            '\0' semaphore.

#ifndef MBSHADER_CML_ERROR_H
#define MBSHADER_CML_ERROR_H

//#define CML_KERNEL_ERROR_HANDLING // Is automatically set for _DEBUG builds

//#############################################################################
//###   Error macros
//#############################################################################
#ifdef CML_KERNEL_ERROR_HANDLING

#define STRINGIZE(a)    DO_STRINGIZE(a)
#define DO_STRINGIZE(a) #a

void error_string(RWByteAddressBuffer l_error_buffer,
                  uint p_char0 , uint p_char1 , uint p_char2 , uint p_char3 , uint p_char4 , uint p_char5 , uint p_char6 , uint p_char7,
                  uint p_char8 , uint p_char9 , uint p_char10, uint p_char11, uint p_char12, uint p_char13, uint p_char14, uint p_char15,
                  uint p_char16, uint p_char17, uint p_char18, uint p_char19, uint p_char20, uint p_char21, uint p_char22, uint p_char23,
                  uint p_char24, uint p_char25, uint p_char26, uint p_char27, uint p_char28, uint p_char29, uint p_char30, uint p_char31
    )
{
    uint l_offset = 64;
    l_error_buffer.Store(l_offset + 0 , p_char0 );
    l_error_buffer.Store(l_offset + 1 , p_char1 );
    l_error_buffer.Store(l_offset + 2 , p_char2 );
    l_error_buffer.Store(l_offset + 3 , p_char3 );
    l_error_buffer.Store(l_offset + 4 , p_char4 );
    l_error_buffer.Store(l_offset + 5 , p_char5 );
    l_error_buffer.Store(l_offset + 6 , p_char6 );
    l_error_buffer.Store(l_offset + 7 , p_char7 );
    l_error_buffer.Store(l_offset + 8 , p_char8 );
    l_error_buffer.Store(l_offset + 9 , p_char9 );
    l_error_buffer.Store(l_offset + 10, p_char10);
    l_error_buffer.Store(l_offset + 11, p_char11);
    l_error_buffer.Store(l_offset + 12, p_char12);
    l_error_buffer.Store(l_offset + 13, p_char13);
    l_error_buffer.Store(l_offset + 14, p_char14);
    l_error_buffer.Store(l_offset + 15, p_char15);
    l_error_buffer.Store(l_offset + 16, p_char16);
    l_error_buffer.Store(l_offset + 17, p_char17);
    l_error_buffer.Store(l_offset + 18, p_char18);
    l_error_buffer.Store(l_offset + 19, p_char19);
    l_error_buffer.Store(l_offset + 20, p_char20);
    l_error_buffer.Store(l_offset + 21, p_char21);
    l_error_buffer.Store(l_offset + 22, p_char22);
    l_error_buffer.Store(l_offset + 23, p_char23);
    l_error_buffer.Store(l_offset + 24, p_char24);
    l_error_buffer.Store(l_offset + 25, p_char25);
    l_error_buffer.Store(l_offset + 26, p_char26);
    l_error_buffer.Store(l_offset + 27, p_char27);
    l_error_buffer.Store(l_offset + 28, p_char28);
    l_error_buffer.Store(l_offset + 29, p_char29);
    l_error_buffer.Store(l_offset + 30, p_char30);
    l_error_buffer.Store(l_offset + 31, p_char31);

    return;

    l_error_buffer.Store(64, 't'); // Works

    uint l_message = 'e';
    l_error_buffer.Store(64, l_message); // Works
}

// Check the error state at the start of a kernel
#define CML_CHECK_KERNEL_ERROR                  RWByteAddressBuffer l_error_buffer = ResourceDescriptorHeap[g_push_constants.m_error_uav];if (l_error_buffer.Load(0) > 0) return
#define CML_CLEAR_KERNEL_ERROR                  l_error_buffer.Store(0, 0)
#define CML_SET_KERNEL_ERROR                    { uint l_error_out; l_error_buffer.InterlockedAdd(0, 1, l_error_out); }
#define CML_SET_ERROR_INT(index, number)        { l_error_buffer.Store((index+1)*4, number); }
#define CML_SET_ERROR_FLOAT(index, number)      { l_error_buffer.Store((index+8)*4, asuint(number)); }

#define CML_SET_ERROR_MESSAGE4(char0, char1, char2, char3)                              error_string(l_error_buffer, char0, char1, char2, char3, '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0')

#define CML_SET_ERROR_MESSAGE8(char0, char1, char2, char3, char4, char5, char6, char7)  error_string(l_error_buffer, char0, char1, char2, char3, char4, char5, char6, char7, '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0')

#define CML_SET_ERROR_MESSAGE16(char0, char1, char2 , char3 , char4 , char5 , char6 , char7, \
                                char8, char9, char10, char11, char12, char13, char14, char15) \
error_string(l_error_buffer, char0, char1, char2 , char3 , char4 , char5 , char6 , char7, char8, char9, char10, char11, char12, char13, char14, char15, '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0')

#define CML_SET_ERROR_MESSAGE24(char0 , char1 , char2 , char3 , char4 , char5 , char6 , char7, \
                                char8 , char9 , char10, char11, char12, char13, char14, char15, \
                                char16, char17, char18, char19, char20, char21, char22, char23) \
error_string(l_error_buffer, char0, char1, char2 , char3 , char4 , char5 , char6 , char7, char8, char9, char10, char11, char12, char13, char14, char15, char16, char17, char18, char19, char20, char21, char22, char23, '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0')

#define CML_SET_ERROR_MESSAGE32(char0 , char1 , char2 , char3 , char4 , char5 , char6 , char7, \
                                char8 , char9 , char10, char11, char12, char13, char14, char15, \
                                char16, char17, char18, char19, char20, char21, char22, char23, \
                                char24, char25, char26, char27, char28, char29, char30, char31) \
error_string(l_error_buffer, char0, char1, char2 , char3 , char4 , char5 , char6 , char7, char8, char9, char10, char11, char12, char13, char14, char15, char16, char17, char18, char19, char20, char21, char22, char23, char24, char25, char26, char27, char28, char29, char31, char31)


#else   // CML_KERNEL_ERROR_HANDLING

#define CML_CHECK_KERNEL_ERROR
#define CML_CLEAR_KERNEL_ERROR
#define CML_SET_KERNEL_ERROR
#define CML_SET_ERROR_INT(index, number)
#define CML_SET_ERROR_FLOAT(index, number)

#define CML_SET_ERROR_MESSAGE4( char0, char1, char2, char3)
#define CML_SET_ERROR_MESSAGE8( char0, char1, char2, char3, char4, char5, char6, char7)    error_string(char0, char1, char2, char3, char4, char5, char6, char7, '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0')
#define CML_SET_ERROR_MESSAGE16(char0, char1, char2 , char3 , char4 , char5 , char6 , char7, \
                                char8, char9, char10, char11, char12, char13, char14, char15)
#define CML_SET_ERROR_MESSAGE24(char0 , char1 , char2 , char3 , char4 , char5 , char6 , char7, \
                                char8 , char9 , char10, char11, char12, char13, char14, char15, \
                                char16, char17, char18, char19, char20, char21, char22, char23)
#define CML_SET_ERROR_MESSAGE32(char0 , char1 , char2 , char3 , char4 , char5 , char6 , char7, \
                                char8 , char9 , char10, char11, char12, char13, char14, char15, \
                                char16, char17, char18, char19, char20, char21, char22, char23, \
                                char24, char25, char26, char27, char28, char29, char30, char31)

#endif // CML_KERNEL_ERROR_HANDLING

#endif // MBSHADER_CML_ERROR_H
