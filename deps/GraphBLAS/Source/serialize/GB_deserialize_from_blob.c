//------------------------------------------------------------------------------
// GB_deserialize_from_blob: uncompress a set of blocks from the blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Decompress a single array from a set of compressed blocks in the blob.  If
// the input data is mangled, this method is still safe, since it performs the
// bare minimum sanity checks to ensure no out-of-bounds indexing of arrays.
// However, the contents of output array are not fully checked.  This step is
// done by GB_deserialize, if requested.

#include "GB.h"
#include "serialize/GB_serialize.h"
#include "lz4_wrapper/GB_lz4.h"
#include "zstd_wrapper/GB_zstd.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_FREE_MEMORY (&X, X_mem) ;    \
}

GrB_Info GB_deserialize_from_blob
(
    // output:
    GB_void **X_handle,         // uncompressed output array
    uint64_t *X_mem_handle,     // size of X as allocated
    // input:
    int64_t X_len,              // size of X in bytes
    const GB_void *blob,        // serialized blob of size blob_memsize
    uint64_t blob_memsize,
    uint64_t *Sblocks,          // array of size nblocks
    int32_t nblocks,            // # of compressed blocks for this array
    int32_t method,             // compression method used for each block
    // input/output:
    uint64_t *s_handle          // where to read from the blob
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    int memlane = 0 ;           // FIXME memlane param
    uint64_t mem = GB_mem (memlane, 0) ;

    ASSERT (blob != NULL) ;
    ASSERT (s_handle != NULL) ;
    ASSERT (X_handle != NULL) ;
    ASSERT (X_mem_handle != NULL) ;
    (*X_handle) = NULL ;
    (*X_mem_handle) = mem ;

    //--------------------------------------------------------------------------
    // parse the method
    //--------------------------------------------------------------------------

    int32_t algo, level ;
    GB_serialize_method (&algo, &level, method) ;

    //--------------------------------------------------------------------------
    // allocate the output array
    //--------------------------------------------------------------------------

    uint64_t X_mem = mem ;
    GB_void *X = NULL ;
    if (nblocks == 0)
    {
        // allocate an "empty" block (of 8 bytes) and set it zero
        X = GB_CALLOC_MEMORY (X_len, sizeof (GB_void), &X_mem) ;
    }
    else
    {
        // allocate a block that is filled below
        X = GB_MALLOC_MEMORY (X_len, sizeof (GB_void), &X_mem) ;
    }

    if (X == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // decompress the blocks from the blob
    //--------------------------------------------------------------------------

    size_t s = (*s_handle) ;
    bool ok = true ;

    if (nblocks == 0)
    {

        // nothing else to do for this array
        ;

    }
    else if (algo == GxB_COMPRESSION_NONE)
    {

        //----------------------------------------------------------------------
        // no compression; the array is held in a single block
        //----------------------------------------------------------------------

        if (nblocks != 1 || Sblocks [0] != X_len || s + X_len > blob_memsize)
        { 
            // blob is invalid: guard against an unsafe memcpy
            ok = false ;
        }
        else
        { 
            // copy the blob into the array X.  This is now safe and secure.
            // The contents of X are not yet checked, however.
            GB_memcpy (X, blob + s, X_len, nthreads_max) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // LZ4, LZ4HC, or ZSTD compression
        //----------------------------------------------------------------------

        int nthreads = GB_IMIN (nthreads_max, nblocks) ;
        int32_t blockid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic) \
            reduction(&&:ok)
        for (blockid = 0 ; blockid < nblocks ; blockid++)
        {
            // get the start and end of the compressed and uncompressed blocks
            int64_t kstart, kend ;
            GB_PARTITION (kstart, kend, X_len, blockid, nblocks) ;
            int64_t s_start = (blockid == 0) ? 0 : Sblocks [blockid-1] ;
            int64_t s_end   = Sblocks [blockid] ;
            size_t  s_memsize  = s_end - s_start ;
            size_t  d_memsize  = kend - kstart ;
            // ensure s_start, s_end, kstart, and kend are all valid,
            // to avoid accessing arrays out of bounds, if input is corrupted.
            if (kstart < 0 || kend < 0 || s_start < 0 || s_end < 0 ||
                kstart >= kend || s_start >= s_end || s_memsize > INT32_MAX ||
                s + s_start > blob_memsize || s + s_end > blob_memsize ||
                kstart > X_len || kend > X_len || d_memsize > INT32_MAX)
            { 
                // blob is invalid
                ok = false ;
            }
            else
            { 
                // uncompress the compressed block of size s_memsize
                // from blob [s + s_start:s_end-1] into X [kstart:kend-1].
                // This is safe and secure so far.  The contents of X are
                // not yet checked, however.  That step is done in
                // GB_deserialize, if requested.
                const char *src = (const char *) (blob + s + s_start) ;
                char *dst = (char *) (X + kstart) ;
                if (algo == GxB_COMPRESSION_ZSTD)
                { 
                    // ZSTD
                    size_t u = ZSTD_decompress (dst, d_memsize, src, s_memsize);
                    if (u != d_memsize)
                    {
                        // blob is invalid
                        ok = false ;
                    }
                }
                else
                { 
                    // LZ4 or LZ4HC
                    int src_memsize = (int) s_memsize ;
                    int dst_memsize = (int) d_memsize ;
                    int u = LZ4_decompress_safe (src, dst,
                        src_memsize, dst_memsize) ;
                    if (u != dst_memsize)
                    {
                        // blob is invalid
                        ok = false ;
                    }
                }
            }
        }
    }

    if (!ok)
    { 
        // decompression failure; blob is invalid
        GB_FREE_ALL ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // return result: X, its size, and updated index into the blob
    //--------------------------------------------------------------------------

    (*X_handle) = X ;
    (*X_mem_handle) = X_mem ;
    if (nblocks > 0)
    { 
        s += Sblocks [nblocks-1] ;
    }
    (*s_handle) = s ;
    return (GrB_SUCCESS) ;
}

