//------------------------------------------------------------------------------
// GB_lz4.h: definitions for a wrapper for the LZ4 compression library
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// It's possible that the user application has its own copy of the LZ4 library,
// which wouldn't be using the SuiteSparse:GraphBLAS memory allocator.  To
// avoid any conflict between multiple copies of the LZ4 library, all global
// symbols LZ4_* are renamed GBLZ4 (LZ4_*), via #defines below.

#ifndef GB_LZ4_H 
#define GB_LZ4_H 

// LZ4 has its own GB macro, so #undefine the GraphBLAS one, and use GBLZ4
// to rename the LZ4 functions.
#undef GB

#ifdef GBMATLAB
    #define GBLZ4(x) GB_EVAL2 (GM_, x)
#else
    #define GBLZ4(x) GB_EVAL2 (GB_, x)
#endif

//------------------------------------------------------------------------------
// methods called directly by GraphBLAS
//------------------------------------------------------------------------------

// Note that all size parameters are ints, which are assumed to be int32_t.
// The largest block that LZ4 can compress is LZ4_MAX_INPUT_SIZE, which is just
// under 2GB.

// int LZ4_compressBound (int s) : returns the maximum size for the compression
// of a block of s bytes, which is (s + s/255 + 16), or 0 if s is negative or
// too large.  s must be less than LZ4_MAX_INPUT_SIZE.
#define LZ4_compressBound       GBLZ4 (LZ4_compressBound)

// int LZ4_compress_default (const char *src, char *dst, int srcSize, int
// dstCap) : compresses the uncompressed src block of size srcSize into the
// output buffer dst of size dstCap.  Returns the # of bytes written to dst, or
// zero if compression fails.
#define LZ4_compress_default    GBLZ4 (LZ4_compress_default)

// int LZ4_decompress_safe (const char *src, char *dst, int compSize,
// int dstCap) : decompresses a compressed src block of size compSize into the
// dst block of size dstCap.  Returns the # of bytes written to dst, or zero if
// decompression fails.
#define LZ4_decompress_safe     GBLZ4 (LZ4_decompress_safe)

// int LZ4_compress_HC (const char *src, char *dst, int srcSize, int dstCaop,
// int level) : same arguments as LZ4_compress_default, except with the added
// level parameter.
#define LZ4_compress_HC         GBLZ4 (LZ4_compress_HC)

//------------------------------------------------------------------------------
// ensure that LZ4_malloc, LZ4_calloc, and LZ4_free are used.
//------------------------------------------------------------------------------

// LZ4 will use these 3 functions in place of malloc, calloc, and free.  They
// are defined in GB_lz4.c, and rely on the malloc and free methods provided by
// the user application to GraphBLAS by GrB_init or GxB_init.

#define LZ4_USER_MEMORY_FUNCTIONS
#define LZ4_malloc  GBLZ4 (LZ4_malloc)
#define LZ4_calloc  GBLZ4 (LZ4_calloc)
#define LZ4_free    GBLZ4 (LZ4_free)
void *LZ4_malloc (size_t s) ;
void *LZ4_calloc (size_t n, size_t s) ;
void  LZ4_free (void *p) ;

//------------------------------------------------------------------------------
// methods not directly used, or not used at all by GraphBLAS
//------------------------------------------------------------------------------

// LZ4 methods:
#define LZ4_attach_dictionary                   \
 GBLZ4 (LZ4_attach_dictionary)
#define LZ4_compress                            \
 GBLZ4 (LZ4_compress)
#define LZ4_compress_continue                   \
 GBLZ4 (LZ4_compress_continue)
#define LZ4_compress_destSize                   \
 GBLZ4 (LZ4_compress_destSize)
#define LZ4_compress_fast                       \
 GBLZ4 (LZ4_compress_fast)
#define LZ4_compress_fast_continue              \
 GBLZ4 (LZ4_compress_fast_continue)
#define LZ4_compress_fast_extState              \
 GBLZ4 (LZ4_compress_fast_extState)
#define LZ4_compress_fast_extState_fastReset    \
 GBLZ4 (LZ4_compress_fast_extState_fastReset)
#define LZ4_compress_forceExtDict               \
 GBLZ4 (LZ4_compress_forceExtDict)
#define LZ4_compress_limitedOutput              \
 GBLZ4 (LZ4_compress_limitedOutput)
#define LZ4_compress_limitedOutput_continue     \
 GBLZ4 (LZ4_compress_limitedOutput_continue)
#define LZ4_compress_limitedOutput_withState    \
 GBLZ4 (LZ4_compress_limitedOutput_withState)
#define LZ4_compress_withState                  \
 GBLZ4 (LZ4_compress_withState)
#define LZ4_create                              \
 GBLZ4 (LZ4_create)
#define LZ4_createStream                        \
 GBLZ4 (LZ4_createStream)
#define LZ4_createStreamDecode                  \
 GBLZ4 (LZ4_createStreamDecode)
#define LZ4_decoderRingBufferSize               \
 GBLZ4 (LZ4_decoderRingBufferSize)
#define LZ4_decompress_fast                     \
 GBLZ4 (LZ4_decompress_fast)
#define LZ4_decompress_fast_continue            \
 GBLZ4 (LZ4_decompress_fast_continue)
#define LZ4_decompress_fast_usingDict           \
 GBLZ4 (LZ4_decompress_fast_usingDict)
#define LZ4_decompress_fast_withPrefix64k       \
 GBLZ4 (LZ4_decompress_fast_withPrefix64k)
#define LZ4_decompress_safe_continue            \
 GBLZ4 (LZ4_decompress_safe_continue)
#define LZ4_decompress_safe_forceExtDict        \
 GBLZ4 (LZ4_decompress_safe_forceExtDict)
#define LZ4_decompress_safe_partial             \
 GBLZ4 (LZ4_decompress_safe_partial)
#define LZ4_decompress_safe_usingDict           \
 GBLZ4 (LZ4_decompress_safe_usingDict)
#define LZ4_decompress_safe_withPrefix64k       \
 GBLZ4 (LZ4_decompress_safe_withPrefix64k)
#define LZ4_freeStream                          \
 GBLZ4 (LZ4_freeStream)
#define LZ4_freeStreamDecode                    \
 GBLZ4 (LZ4_freeStreamDecode)
#define LZ4_initStream                          \
 GBLZ4 (LZ4_initStream)
#define LZ4_loadDict                            \
 GBLZ4 (LZ4_loadDict)
#define LZ4_resetStream                         \
 GBLZ4 (LZ4_resetStream)
#define LZ4_resetStream_fast                    \
 GBLZ4 (LZ4_resetStream_fast)
#define LZ4_resetStreamState                    \
 GBLZ4 (LZ4_resetStreamState)
#define LZ4_saveDict                            \
 GBLZ4 (LZ4_saveDict)
#define LZ4_setStreamDecode                     \
 GBLZ4 (LZ4_setStreamDecode)
#define LZ4_sizeofState                         \
 GBLZ4 (LZ4_sizeofState)
#define LZ4_sizeofStreamState                   \
 GBLZ4 (LZ4_sizeofStreamState)
#define LZ4_slideInputBuffer                    \
 GBLZ4 (LZ4_slideInputBuffer)
#define LZ4_uncompress                          \
 GBLZ4 (LZ4_uncompress)
#define LZ4_uncompress_unknownOutputSize        \
 GBLZ4 (LZ4_uncompress_unknownOutputSize)
#define LZ4_versionNumber                       \
 GBLZ4 (LZ4_versionNumber)
#define LZ4_versionString                       \
 GBLZ4 (LZ4_versionString)

// LZ4HC methods:
#define LZ4_attach_HC_dictionary                    \
 GBLZ4 (LZ4_attach_HC_dictionary)
#define LZ4_compressHC                              \
 GBLZ4 (LZ4_compressHC)
#define LZ4_compressHC2                             \
 GBLZ4 (LZ4_compressHC2)
#define LZ4_compressHC2_continue                    \
 GBLZ4 (LZ4_compressHC2_continue)
#define LZ4_compressHC2_limitedOutput               \
 GBLZ4 (LZ4_compressHC2_limitedOutput)
#define LZ4_compressHC2_limitedOutput_continue      \
 GBLZ4 (LZ4_compressHC2_limitedOutput_continue)
#define LZ4_compressHC2_limitedOutput_withStateHC   \
 GBLZ4 (LZ4_compressHC2_limitedOutput_withStateHC)
#define LZ4_compressHC2_withStateHC                 \
 GBLZ4 (LZ4_compressHC2_withStateHC)
#define LZ4_compress_HC_continue                    \
 GBLZ4 (LZ4_compress_HC_continue)
#define LZ4_compressHC_continue                     \
 GBLZ4 (LZ4_compressHC_continue)
#define LZ4_compress_HC_continue_destSize           \
 GBLZ4 (LZ4_compress_HC_continue_destSize)
#define LZ4_compress_HC_destSize                    \
 GBLZ4 (LZ4_compress_HC_destSize)
#define LZ4_compress_HC_extStateHC                  \
 GBLZ4 (LZ4_compress_HC_extStateHC)
#define LZ4_compress_HC_extStateHC_fastReset        \
 GBLZ4 (LZ4_compress_HC_extStateHC_fastReset)
#define LZ4_compressHC_limitedOutput                \
 GBLZ4 (LZ4_compressHC_limitedOutput)
#define LZ4_compressHC_limitedOutput_continue       \
 GBLZ4 (LZ4_compressHC_limitedOutput_continue)
#define LZ4_compressHC_limitedOutput_withStateHC    \
 GBLZ4 (LZ4_compressHC_limitedOutput_withStateHC)
#define LZ4_compressHC_withStateHC                  \
 GBLZ4 (LZ4_compressHC_withStateHC)
#define LZ4_createHC                                \
 GBLZ4 (LZ4_createHC)
#define LZ4_createStreamHC                          \
 GBLZ4 (LZ4_createStreamHC)
#define LZ4_favorDecompressionSpeed                 \
 GBLZ4 (LZ4_favorDecompressionSpeed)
#define LZ4_freeHC                                  \
 GBLZ4 (LZ4_freeHC)
#define LZ4_freeStreamHC                            \
 GBLZ4 (LZ4_freeStreamHC)
#define LZ4_initStreamHC                            \
 GBLZ4 (LZ4_initStreamHC)
#define LZ4_loadDictHC                              \
 GBLZ4 (LZ4_loadDictHC)
#define LZ4_resetStreamHC                           \
 GBLZ4 (LZ4_resetStreamHC)
#define LZ4_resetStreamHC_fast                      \
 GBLZ4 (LZ4_resetStreamHC_fast)
#define LZ4_resetStreamStateHC                      \
 GBLZ4 (LZ4_resetStreamStateHC)
#define LZ4_saveDictHC                              \
 GBLZ4 (LZ4_saveDictHC)
#define LZ4_setCompressionLevel                     \
 GBLZ4 (LZ4_setCompressionLevel)
#define LZ4_sizeofStateHC                           \
 GBLZ4 (LZ4_sizeofStateHC)
#define LZ4_sizeofStreamStateHC                     \
 GBLZ4 (LZ4_sizeofStreamStateHC)
#define LZ4_slideInputBufferHC                      \
 GBLZ4 (LZ4_slideInputBufferHC)

//------------------------------------------------------------------------------
// disable LZ4 deprecation warnings and include all LZ4 definitions  
//------------------------------------------------------------------------------

// GraphBLAS does not use deprecated functions, but the warnings pop up anyway
// when GraphBLAS is built, so silence them with this #define:
#define LZ4_DISABLE_DEPRECATE_WARNINGS
// #include "lz4.h"
#include "lz4hc.h"
#endif

