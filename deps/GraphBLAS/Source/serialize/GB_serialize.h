//------------------------------------------------------------------------------
// GB_serialize.h: definitions for GB_serialize_* and deserialize methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SERIALIZE_H
#define GB_SERIALIZE_H

GrB_Info GB_serialize               // serialize a matrix into a blob
(
    // output:
    GB_void **blob_handle,          // serialized matrix, allocated on output
    uint64_t *blob_memsize_handle,  // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    int32_t method,                 // method to use
    GB_Werk Werk
) ;

void GB_serialize_method
(
    // output
    int32_t *algo,                  // algorithm to use
    int32_t *level,                 // compression level
    // input
    int32_t method
) ;

GrB_Info GB_deserialize             // deserialize a matrix from a blob
(
    // output:
    GrB_Matrix *Chandle,            // output matrix created from the blob
    // input:
    GrB_Type type_expected,         // type expected (NULL for any built-in)
    const GB_void *blob,            // serialized matrix 
    uint64_t blob_memsize           // size of the blob
) ;

typedef struct
{
    void *p ;                       // pointer to the compressed block
    uint64_t p_mem ;                // allocated size of compressed block
}
GB_blocks ;

GrB_Info GB_serialize_array
(
    // output:
    GB_blocks **Blocks_handle,          // Blocks: array of size nblocks+1
    uint64_t *Blocks_mem_handle,        // size of Blocks
    uint64_t **Sblocks_handle,          // Sblocks: array of size nblocks+1
    uint64_t *Sblocks_mem_handle,       // size of Sblocks
    int32_t *nblocks_handle,            // # of blocks
    int32_t *method_used,               // method used
    uint64_t *compressed_memsize,       // size of compressed block, or upper
                                        // bound if dryrun is true
    // input:
    bool dryrun,                        // if true, just esimate the size
    GB_void *X,                         // input array of size len
    int64_t len,                        // size of X, in bytes
    int32_t method,                     // compression method requested
    int32_t algo,                       // compression algorithm
    int32_t level,                      // compression level
    GB_Werk Werk
) ;

void GB_serialize_free_blocks
(
    GB_blocks **Blocks_handle,      // array of size nblocks
    uint64_t Blocks_mem,            // memsize and memlane of Blocks
    int32_t nblocks                 // # of blocks, or zero if no blocks
) ;

void GB_serialize_to_blob
(
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    uint64_t *s_handle,     // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks+1
    uint64_t *Sblocks,      // array of size nblocks
    int32_t nblocks,        // # of blocks
    int nthreads_max        // # of threads to use
) ;

GrB_Info GB_deserialize_from_blob
(
    // output:
    GB_void **X_handle,         // uncompressed output array
    uint64_t *X_mem_handle,     // memsize and memlane of X as allocated
    // input:
    int64_t X_len,              // size of X in bytes
    const GB_void *blob,        // serialized blob of size blob_memsize
    uint64_t blob_memsize,
    uint64_t *Sblocks,          // array of size nblocks
    int32_t nblocks,            // # of compressed blocks for this array
    int32_t method_used,        // compression method used for each block
    // input/output:
    uint64_t *s_handle            // where to read from the blob
) ;

#define GB_BLOB_HEADER_SIZE \
    sizeof (uint64_t)           /* blob_memsize                         */  \
    + 11 * sizeof (int64_t)     /* vlen, vdim, nvec, nvec_nonempty,     */  \
                                /* nvals, typesize, A[phbix]_len        */  \
    + 14 * sizeof (int32_t)     /* version, typecode, sparsity_control, */  \
                                /* A[phbix]_nblocks, A[phbix]_method,   */  \
                                /* sparsity_iso_csc                     */  \
    + 2 * sizeof (float)        /* hyper_switch, bitmap_switch          */

// write a scalar to the blob
#define GB_BLOB_WRITE(x,type)                                               \
    memcpy (blob + s, &(x), sizeof (type)) ;                                \
    s += sizeof (type) ;

// write a uint64_t array Sblocks[1:n] to the blob, of size n+1, but do not
// write the first entry (so only n words are written)
#define GB_BLOB_WRITES(S,n) \
    if (n > 0)                                                              \
    {                                                                       \
        memcpy (((GB_void *) blob) + s, S + 1, n * sizeof (uint64_t)) ;     \
        s += n * sizeof (uint64_t) ;                                        \
    }

// read a scalar from the blob
#define GB_BLOB_READ(x,type)                                                \
    type x ;                                                                \
    memcpy (&x, ((GB_void *) blob) + s, sizeof (type)) ;                    \
    s += sizeof (type) ;

// get a uint64_t pointer to an array in the blob, of size n
#define GB_BLOB_READS(S,n)                                                  \
    uint64_t *S = (uint64_t *) (blob + s) ;                                 \
    s += n * sizeof (uint64_t) ;

static inline uint32_t GB_pji_control_encoding (int8_t control)
{
    switch (control)
    {
        default:
        case 0  : return (0) ;
        case 32 : return (1) ;
        case 64 : return (2) ;
    }
}

static inline int8_t GB_pji_control_decoding (uint32_t encoding)
{
    switch (encoding)
    {
        default:
        case 0  : return (0) ;
        case 1  : return (32) ;
        case 2  : return (64) ;
    }
}

#endif

