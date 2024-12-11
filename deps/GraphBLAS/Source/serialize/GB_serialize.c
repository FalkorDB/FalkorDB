//------------------------------------------------------------------------------
// GB_serialize: compress and serialize a GrB_Matrix into a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel compression method for a GrB_Matrix.  The input matrix may have
// shallow components; the output is unaffected by this.  The output blob is
// allocated on output (for GxB_Matrix_serialize) or used pre-allocated on
// input (for GrB_Matrix_serialize).  This method also does a dry run to
// estimate the size of the blob for GrB_Matrix_serializeSize.

#include "GB.h"
#include "get_set/GB_get_set.h"
#include "serialize/GB_serialize.h"

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_FREE (&Ap_Sblocks, Ap_Sblocks_size) ;    \
    GB_FREE (&Ah_Sblocks, Ah_Sblocks_size) ;    \
    GB_FREE (&Ab_Sblocks, Ab_Sblocks_size) ;    \
    GB_FREE (&Ai_Sblocks, Ai_Sblocks_size) ;    \
    GB_FREE (&Ax_Sblocks, Ax_Sblocks_size) ;    \
    GB_serialize_free_blocks (&Ap_Blocks, Ap_Blocks_size, Ap_nblocks) ; \
    GB_serialize_free_blocks (&Ah_Blocks, Ah_Blocks_size, Ah_nblocks) ; \
    GB_serialize_free_blocks (&Ab_Blocks, Ab_Blocks_size, Ab_nblocks) ; \
    GB_serialize_free_blocks (&Ai_Blocks, Ai_Blocks_size, Ai_nblocks) ; \
    GB_serialize_free_blocks (&Ax_Blocks, Ax_Blocks_size, Ax_nblocks) ; \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    if (!preallocated_blob)                     \
    {                                           \
        GB_FREE (&blob, blob_size_allocated) ;  \
    }                                           \
}

GrB_Info GB_serialize               // serialize a matrix into a blob
(
    // output:
    GB_void **blob_handle,          // serialized matrix, allocated on output
                                    // for GxB_Matrix_serialize, or provided by
                                    // GrB_Matrix_serialize.  NULL for
                                    // GrB_Matrix_serialize_size.
    size_t *blob_size_handle,       // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    int32_t method,                 // method to use
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob_size_handle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for serialize", GB0) ;

    //--------------------------------------------------------------------------
    // determine what serialization to do
    //--------------------------------------------------------------------------

    GB_void *blob = NULL ;
    size_t blob_size_allocated = 0 ;
    bool dryrun = false ;
    bool preallocated_blob = false ;
    if (blob_handle == NULL)
    { 
        // for GrB_Matrix_serializeSize:  the blob is not provided on input,
        // and not allocated.  Just compute an upper bound only.
        dryrun = true ;
    }
    else if (*blob_handle != NULL)
    { 
        // for GrB_Matrix_serialize:  the blob is already allocated by the user
        // and provided on input.  Fill the blob, and return the blob size as
        // the # of bytes written to the blob.
        preallocated_blob = true ;
        blob = (*blob_handle) ;
        blob_size_allocated = (*blob_size_handle) ;
    }
    else
    { 
        // for GxB_Matrix_serialize:  the blob is not allocated yet. Allocate
        // it and return it below, and return the blob size.
    }

    (*blob_size_handle) = 0 ;

    GB_blocks *Ap_Blocks = NULL ; size_t Ap_Blocks_size = 0 ;
    GB_blocks *Ah_Blocks = NULL ; size_t Ah_Blocks_size = 0 ;
    GB_blocks *Ab_Blocks = NULL ; size_t Ab_Blocks_size = 0 ;
    GB_blocks *Ai_Blocks = NULL ; size_t Ai_Blocks_size = 0 ;
    GB_blocks *Ax_Blocks = NULL ; size_t Ax_Blocks_size = 0 ;
    int64_t *Ap_Sblocks = NULL  ; size_t Ap_Sblocks_size = 0 ;
    int64_t *Ah_Sblocks = NULL  ; size_t Ah_Sblocks_size = 0 ;
    int64_t *Ab_Sblocks = NULL  ; size_t Ab_Sblocks_size = 0 ;
    int64_t *Ai_Sblocks = NULL  ; size_t Ai_Sblocks_size = 0 ;
    int64_t *Ax_Sblocks = NULL  ; size_t Ax_Sblocks_size = 0 ;
    int32_t Ap_nblocks = 0      ; size_t Ap_compressed_size = 0 ;
    int32_t Ah_nblocks = 0      ; size_t Ah_compressed_size = 0 ;
    int32_t Ab_nblocks = 0      ; size_t Ab_compressed_size = 0 ;
    int32_t Ai_nblocks = 0      ; size_t Ai_compressed_size = 0 ;
    int32_t Ax_nblocks = 0      ; size_t Ax_compressed_size = 0 ;

    //--------------------------------------------------------------------------
    // ensure all pending work is finished
    //--------------------------------------------------------------------------

    GB_OK (GB_wait (A, "A to serialize", Werk)) ;
    ASSERT (A->nvec_nonempty >= 0) ;

    //--------------------------------------------------------------------------
    // determine maximum # of threads
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // parse the method
    //--------------------------------------------------------------------------

    int32_t algo, level ;
    GB_serialize_method (&algo, &level, method) ;
    method = algo + level ;
    GBURBLE ("(compression: %s%s%s%s:%d) ",
        (algo == GxB_COMPRESSION_NONE ) ? "none" : "",
        (algo == GxB_COMPRESSION_LZ4  ) ? "LZ4" : "",
        (algo == GxB_COMPRESSION_LZ4HC) ? "LZ4HC" : "",
        (algo == GxB_COMPRESSION_ZSTD ) ? "ZSTD" : "",
        level) ;

    //--------------------------------------------------------------------------
    // get the content of the matrix
    //--------------------------------------------------------------------------

    int32_t version = GxB_IMPLEMENTATION ;
    int64_t vlen = A->vlen ;
    int64_t vdim = A->vdim ;
    int64_t nvec = A->nvec ;
    int64_t nvals = A->nvals ;
    int64_t nvec_nonempty = A->nvec_nonempty ;
    int32_t sparsity = GB_sparsity (A) ;
    bool iso = A->iso ;
    float hyper_switch = A->hyper_switch ;
    float bitmap_switch = A->bitmap_switch ;
    int32_t sparsity_control = A->sparsity_control ;
    // the matrix has no pending work
    ASSERT (A->Pending == NULL) ;
    ASSERT (A->nzombies == 0) ;
    ASSERT (!A->jumbled) ;
    GrB_Type atype = A->type ;
    int64_t typesize = atype->size ;
    int32_t typecode = (int32_t) (atype->code) ;
    int64_t anz = GB_nnz (A) ;
    int64_t anz_held = GB_nnz_held (A) ;

    // determine the uncompressed sizes of Ap, Ah, Ab, Ai, and Ax
    int64_t Ap_len = 0 ;
    int64_t Ah_len = 0 ;
    int64_t Ab_len = 0 ;
    int64_t Ai_len = 0 ;
    int64_t Ax_len = 0 ;
    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            Ah_len = sizeof (GrB_Index) * nvec ;
            // fall through to the sparse case
        case GxB_SPARSE :
            Ap_len = sizeof (GrB_Index) * (nvec+1) ;
            Ai_len = sizeof (GrB_Index) * anz ;
            Ax_len = typesize * (iso ? 1 : anz) ;
            break ;
        case GxB_BITMAP : 
            Ab_len = sizeof (int8_t) * anz_held ;
            // fall through to the full case
        case GxB_FULL : 
            Ax_len = typesize * (iso ? 1 : anz_held) ;
            break ;
        default: ;
    }

    //--------------------------------------------------------------------------
    // compress each array (Ap, Ah, Ab, Ai, and Ax)
    //--------------------------------------------------------------------------

    // For the dryrun case, this just computes A[phbix]_compressed_size as an
    // upper bound on each array size when compressed, and A[phbix]_nblocks.

    int32_t Ap_method, Ah_method, Ab_method, Ai_method, Ax_method ;

    GB_OK (GB_serialize_array (&Ap_Blocks, &Ap_Blocks_size,
        &Ap_Sblocks, &Ap_Sblocks_size, &Ap_nblocks, &Ap_method,
        &Ap_compressed_size, dryrun,
        (GB_void *) A->p, Ap_len, method, algo, level, Werk)) ;

    GB_OK (GB_serialize_array (&Ah_Blocks, &Ah_Blocks_size,
        &Ah_Sblocks, &Ah_Sblocks_size, &Ah_nblocks, &Ah_method,
        &Ah_compressed_size, dryrun,
        (GB_void *) A->h, Ah_len, method, algo, level, Werk)) ;

    GB_OK (GB_serialize_array (&Ab_Blocks, &Ab_Blocks_size,
        &Ab_Sblocks, &Ab_Sblocks_size, &Ab_nblocks, &Ab_method,
        &Ab_compressed_size, dryrun,
        (GB_void *) A->b, Ab_len, method, algo, level, Werk)) ;

    GB_OK (GB_serialize_array (&Ai_Blocks, &Ai_Blocks_size,
        &Ai_Sblocks, &Ai_Sblocks_size, &Ai_nblocks, &Ai_method,
        &Ai_compressed_size, dryrun,
        (GB_void *) A->i, Ai_len, method, algo, level, Werk)) ;

    GB_OK (GB_serialize_array (&Ax_Blocks, &Ax_Blocks_size,
        &Ax_Sblocks, &Ax_Sblocks_size, &Ax_nblocks, &Ax_method,
        &Ax_compressed_size, dryrun,
        (GB_void *) A->x, Ax_len, method, algo, level, Werk)) ;

    //--------------------------------------------------------------------------
    // determine the size of the blob
    //--------------------------------------------------------------------------

    size_t s =
        // header information
        GB_BLOB_HEADER_SIZE
        // Sblocks for each array
        + Ap_nblocks * sizeof (int64_t)     // Ap_Sblocks [1:Ap_nblocks]
        + Ah_nblocks * sizeof (int64_t)     // Ah_Sblocks [1:Ah_nblocks]
        + Ab_nblocks * sizeof (int64_t)     // Ab_Sblocks [1:Ab_nblocks]
        + Ai_nblocks * sizeof (int64_t)     // Ai_Sblocks [1:Ai_nblocks]
        + Ax_nblocks * sizeof (int64_t)     // Ax_Sblocks [1:Ax_nblocks]
        // type_name for user-defined types
        + ((typecode == GB_UDT_code) ? GxB_MAX_NAME_LEN : 0) ;

    // size of compressed arrays Ap, Ah, Ab, Ai, and Ax in the blob
    s += Ap_compressed_size ;
    s += Ah_compressed_size ;
    s += Ab_compressed_size ;
    s += Ai_compressed_size ;
    s += Ax_compressed_size ;

    // size of the GrB_NAME and GrB_EL_TYPE_STRING, including one nul byte each
    char *user_name = A->user_name ;
    size_t user_name_len = (user_name == NULL) ? 0 : strlen (user_name) ;
    const char *eltype_string = GB_type_name_get (A->type) ;
    size_t eltype_string_len = (eltype_string == NULL) ? 0 :
        strlen (eltype_string) ;
    s += (user_name_len + 1) + (eltype_string_len + 1) ;

    //--------------------------------------------------------------------------
    // return the upper bound estimate of the blob size, for dryrun
    //--------------------------------------------------------------------------

    if (dryrun)
    { 
        // GrB_Matrix_serializeSize: this is an upper bound on the required
        // size of the blob, not the actual size.
        (*blob_size_handle) = s ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // allocate the blob
    //--------------------------------------------------------------------------

    size_t blob_size_required = s ;     // the exact size required

    if (preallocated_blob)
    {
        // GrB_Matrix_serialize passes in a preallocated blob.
        // Check if it is large enough for the actual blob, of size s.
        if (blob_size_allocated < blob_size_required)
        { 
            // blob too small.  The required minimum size of the blob
            // (blob_size_required) could be returned to the caller.
            GB_FREE_ALL ;
            return (GrB_INSUFFICIENT_SPACE) ;
        }
    }
    else
    {
        // GxB_Matrix_serialize: allocate the block.  The memory pool may
        // increase the blob from size blob_size_required bytes to
        // blob_size_allocated.
        blob = GB_MALLOC (blob_size_required, GB_void, &blob_size_allocated) ;
        if (blob == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        ASSERT (blob_size_allocated >= blob_size_required) ;
    }

    //--------------------------------------------------------------------------
    // write the header and type_name into the blob
    //--------------------------------------------------------------------------

    // 160 bytes, plus 128 bytes for user-defined types

    s = 0 ;
    int32_t sparsity_iso_csc = (4 * sparsity) + (iso ? 2 : 0) +
        (A->is_csc ? 1 : 0) ;

    // size_t is 32 bits if GraphBLAS is compiled in ILP32 mode,
    // so write a 64-bit blob size, regardless of the size of size_t
    uint64_t blob_size_required64 = (uint64_t) blob_size_required ;
    GB_BLOB_WRITE (blob_size_required64, uint64_t) ;

    GB_BLOB_WRITE (typecode, int32_t) ;
    GB_BLOB_WRITE (version, int32_t) ;
    GB_BLOB_WRITE (vlen, int64_t) ;
    GB_BLOB_WRITE (vdim, int64_t) ;
    GB_BLOB_WRITE (nvec, int64_t) ;
    GB_BLOB_WRITE (nvec_nonempty, int64_t) ;
    GB_BLOB_WRITE (nvals, int64_t) ;
    GB_BLOB_WRITE (typesize, int64_t) ;
    GB_BLOB_WRITE (Ap_len, int64_t) ;
    GB_BLOB_WRITE (Ah_len, int64_t) ;
    GB_BLOB_WRITE (Ab_len, int64_t) ;
    GB_BLOB_WRITE (Ai_len, int64_t) ;
    GB_BLOB_WRITE (Ax_len, int64_t) ;
    GB_BLOB_WRITE (hyper_switch, float) ;
    GB_BLOB_WRITE (bitmap_switch, float) ;
    GB_BLOB_WRITE (sparsity_control, int32_t) ;
    GB_BLOB_WRITE (sparsity_iso_csc, int32_t);
    GB_BLOB_WRITE (Ap_nblocks, int32_t) ; GB_BLOB_WRITE (Ap_method, int32_t) ;
    GB_BLOB_WRITE (Ah_nblocks, int32_t) ; GB_BLOB_WRITE (Ah_method, int32_t) ;
    GB_BLOB_WRITE (Ab_nblocks, int32_t) ; GB_BLOB_WRITE (Ab_method, int32_t) ;
    GB_BLOB_WRITE (Ai_nblocks, int32_t) ; GB_BLOB_WRITE (Ai_method, int32_t) ;
    GB_BLOB_WRITE (Ax_nblocks, int32_t) ; GB_BLOB_WRITE (Ax_method, int32_t) ;

    // 128 bytes, if present
    if (typecode == GB_UDT_code)
    { 
        // only copy the type_name for user-defined types
        memset (blob + s, 0, GxB_MAX_NAME_LEN) ;
        #if GB_COMPILER_GCC
        #if (__GNUC__ > 5)
        #pragma GCC diagnostic ignored "-Wstringop-truncation"
        #endif
        #endif
        strncpy ((char *) (blob + s), atype->name, GxB_MAX_NAME_LEN-1) ;
        s += GxB_MAX_NAME_LEN ;
    }

    //--------------------------------------------------------------------------
    // copy the compressed arrays into the blob
    //--------------------------------------------------------------------------

    // 8 * (# blocks for Ap, Ah, Ab, Ai, Ax)
    GB_BLOB_WRITES (Ap_Sblocks, Ap_nblocks) ;
    GB_BLOB_WRITES (Ah_Sblocks, Ah_nblocks) ;
    GB_BLOB_WRITES (Ab_Sblocks, Ab_nblocks) ;
    GB_BLOB_WRITES (Ai_Sblocks, Ai_nblocks) ;
    GB_BLOB_WRITES (Ax_Sblocks, Ax_nblocks) ;

    GB_serialize_to_blob (blob, &s, Ap_Blocks, Ap_Sblocks+1, Ap_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ah_Blocks, Ah_Sblocks+1, Ah_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ab_Blocks, Ab_Sblocks+1, Ab_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ai_Blocks, Ai_Sblocks+1, Ai_nblocks,
        nthreads_max) ;
    GB_serialize_to_blob (blob, &s, Ax_Blocks, Ax_Sblocks+1, Ax_nblocks,
        nthreads_max) ;

    //--------------------------------------------------------------------------
    // append the GrB_NAME and GrB_EL_TYPE_STRING to the blob
    //--------------------------------------------------------------------------

    if (user_name != NULL)
    { 
        // write the GrB_NAME of the matrix (including the nul byte)
        strcpy ((char *) (blob + s), user_name) ;
        s += user_name_len ;
    }
    blob [s++] = 0 ;    // terminate the GrB_NAME with a nul byte

    if (eltype_string != NULL)
    { 
        // write the EL_TYPE_STRING of the matrix type (including the nul byte)
        strcpy ((char *) (blob + s), eltype_string) ;
        s += eltype_string_len ;
    }
    blob [s++] = 0 ;    // terminate the GrB_EL_TYPE_STRING with a nul byte

    ASSERT (s == blob_size_required) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (!preallocated_blob)
    { 
        // GxB_Matrix_serialize: giving the blob to the user; remove it from
        // the list of malloc'd blocks
        #ifdef GB_MEMDUMP
        printf ("removing blob %p size %ld from memtable\n", blob,  // MEMDUMP
            blob_size_allocated) ;
        #endif
        GB_Global_memtable_remove (blob) ;
        (*blob_handle) = blob ;
    }

    // Return the required size of the blob to the user, not the actual
    // allocated space of the blob.  The latter may be larger because of the
    // memory pool.

    (*blob_size_handle) = blob_size_required ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

