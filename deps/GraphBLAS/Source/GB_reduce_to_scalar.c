//------------------------------------------------------------------------------
// GB_reduce_to_scalar: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// c = accum (c, reduce_to_scalar(A)), reduce entries in a matrix to a scalar.
// Does the work for GrB_*_reduce_TYPE, both matrix and vector.

// This function does not need to know if A is hypersparse or not, and its
// result is the same if A is in CSR or CSC format.

// This function is the only place in all of GraphBLAS where the identity value
// of a monoid is required, but only in one special case: it is required to be
// the return value of c when A has no entries.  The identity value is also
// used internally, in the parallel methods below, to initialize a scalar value
// in each task.  The methods could be rewritten to avoid the use of the
// identity value.  Since this function requires it anyway, for the special
// case when nvals(A) is zero, the existence of the identity value makes the
// code a little simpler.

#include "GB_reduce.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_red__include.h"
#endif
#include "GB_monoid_shared_definitions.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_WERK_POP (F, bool) ;         \
    GB_WERK_POP (W, GB_void) ;      \
}

GrB_Info GB_reduce_to_scalar    // z = reduce_to_scalar (A)
(
    void *c,                    // result scalar
    const GrB_Type ctype,       // the type of scalar, c
    const GrB_BinaryOp accum,   // for c = accum(c,z)
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL (c) ;
    GB_WERK_DECLARE (W, GB_void) ;
    GB_WERK_DECLARE (F, bool) ;

    ASSERT_TYPE_OK (ctype, "type of scalar c", GB0) ;
    ASSERT_MONOID_OK (monoid, "monoid for reduce_to_scalar", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for reduce_to_scalar", GB0) ;
    ASSERT_MATRIX_OK (A, "A for reduce_to_scalar", GB0) ;

    // check domains and dimensions for c = accum (c,z)
    GrB_Type ztype = monoid->op->ztype ;
    GB_OK (GB_compatible (ctype, NULL, NULL, false, accum, ztype, Werk)) ;

    // z = monoid (z,A) must be compatible
    if (!GB_Type_compatible (A->type, ztype))
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // assemble any pending tuples; zombies are OK
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT_IF_PENDING (A) ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t asize = A->type->size ;
    int64_t zsize = ztype->size ;
    int64_t anz = GB_nnz_held (A) ;
    ASSERT (anz >= A->nzombies) ;

    // z = identity
    GB_void z [GB_VLA(zsize)] ;
    memcpy (z, monoid->identity, zsize) ;   // required, if nnz(A) is zero

    //--------------------------------------------------------------------------
    // z = reduce_to_scalar (A) on the GPU(s) or CPU
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    #if defined ( SUITESPARSE_CUDA )
    if (GB_reduce_to_scalar_cuda_branch (monoid, A))
    {

        //----------------------------------------------------------------------
        // via the CUDA kernel
        //----------------------------------------------------------------------

        GrB_Matrix V = NULL ;
        info = GB_reduce_to_scalar_cuda (z, &V, monoid, A) ;

        if (V != NULL)
        {
            // reduction must continue.  Result is in V, not the scalar z.
            ASSERT (info == GrB_SUCCESS) ;
            if (V->vlen == 1)
            {
                // the CUDA kernel has reduced A to a single scalar, V [0],
                // with a single threadblock; no more recursion.  Copy the
                // single scalar into z, for the accum phase below.
                memcpy (z, V->x, zsize) ;
                GB_Matrix_free (&V) ;
            }
            else
            {
                // the CUDA kernel has reduced A to an array V; keep going
                info = GB_reduce_to_scalar (c, ctype, accum, monoid, V, Werk) ;
                GB_Matrix_free (&V) ;
                return (info) ;
            }
        }

        // GB_reduce_to_scalar_cuda may refuse to do the reduction and indicate
        // this by returning GrB_NO_VALUE.  If so, the CPU will do it below.
        if (!(info == GrB_SUCCESS || info == GrB_NO_VALUE))
        {
            // GB_reduce_to_scalar_cuda has returned an error
            // (out of memory, or other error)
            return (info) ;
        }

    }
    #endif

    if (info == GrB_NO_VALUE)
    {

        //----------------------------------------------------------------------
        // use OpenMP on the CPU threads
        //----------------------------------------------------------------------

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
        int ntasks = (nthreads == 1) ? 1 : (64 * nthreads) ;
        ntasks = GB_IMIN (ntasks, anz) ;
        ntasks = GB_IMAX (ntasks, 1) ;

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        GB_WERK_PUSH (W, ntasks * zsize, GB_void) ;
        GB_WERK_PUSH (F, ntasks, bool) ;
        if (W == NULL || F == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // z = reduce_to_scalar (A)
        //----------------------------------------------------------------------

        // get terminal value, if any
        GB_void *restrict zterminal = (GB_void *) monoid->terminal ;

        if (anz == A->nzombies)
        { 

            //------------------------------------------------------------------
            // no live entries in A; nothing to do
            //------------------------------------------------------------------

            info = GrB_SUCCESS ;

        }
        else if (A->iso)
        { 

            //------------------------------------------------------------------
            // via the iso kernel
            //------------------------------------------------------------------

            // this takes at most O(log(nvals(A))) time, for any monoid
            GB_reduce_to_scalar_iso (z, monoid, A) ;
            info = GrB_SUCCESS ;

        }
        else if (A->type == ztype)
        { 

            //------------------------------------------------------------------
            // via the factory kernel
            //------------------------------------------------------------------

            #ifndef GBCOMPACT
            GB_IF_FACTORY_KERNELS_ENABLED
            { 

                //--------------------------------------------------------------
                // define the worker for the switch factory
                //--------------------------------------------------------------

                #define GB_red(opname,aname) \
                    GB (_red_ ## opname ## aname)

                #define GB_RED_WORKER(opname,aname,ztype)                   \
                {                                                           \
                    info = GB_red (opname, aname) ((ztype *) z, A, W, F,    \
                        ntasks, nthreads) ;                                 \
                }                                                           \
                break ;

                //--------------------------------------------------------------
                // launch the switch factory
                //--------------------------------------------------------------

                // controlled by opcode and typecode
                GB_Opcode opcode = monoid->op->opcode ;
                GB_Type_code typecode = A->type->code ;
                ASSERT (typecode <= GB_UDT_code) ;

                #include "GB_red_factory.c"
            }
            #endif
        }

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_reduce_to_scalar_jit (z, monoid, A, W, F, ntasks,
                nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        {

            //------------------------------------------------------------------
            // generic worker
            //------------------------------------------------------------------

            #include "GB_generic.h"

            GxB_binary_function freduce = monoid->op->binop_function ;

            // ztype z = identity
            #define GB_DECLARE_IDENTITY(z)                          \
                GB_void z [GB_VLA(zsize)] ;                         \
                memcpy (z, monoid->identity, zsize) ;

            // const zidentity = identity
            #define GB_DECLARE_IDENTITY_CONST(z)                    \
                const GB_void *z = monoid->identity ;

            // const zterminal = terminal_value
            #undef  GB_DECLARE_TERMINAL_CONST
            #define GB_DECLARE_TERMINAL_CONST(zterminal)            \
                const GB_void *zterminal = monoid->terminal ;

            #define GB_A_TYPE GB_void

            // no panel used
            #define GB_PANEL 1
            #define GB_NO_PANEL_CASE

            // W [k] = z, no typecast
            #define GB_COPY_SCALAR_TO_ARRAY(W, k, z)                \
                memcpy (W +(k*zsize), z, zsize)

            // z += W [k], no typecast
            #define GB_ADD_ARRAY_TO_SCALAR(z,W,k)                   \
                freduce (z, z, W +((k)*zsize))

            if (A->type == ztype)
            {

                //--------------------------------------------------------------
                // generic worker: sum up the entries, no typecasting
                //--------------------------------------------------------------

                GB_BURBLE_MATRIX (A, "(generic reduce to scalar: %s) ",
                    monoid->op->name) ;

                // the switch factory didn't handle this case

                // t += (ztype) Ax [p], but no typecasting needed
                #define GB_GETA_AND_UPDATE(t,Ax,p)                      \
                    freduce (t, t, Ax +((p)*zsize))

                if (zterminal == NULL)
                { 
                    // monoid is not terminal
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 0
                    #undef  GB_TERMINAL_CONDITION
                    #define GB_TERMINAL_CONDITION(z,zterminal) 0
                    #undef  GB_IF_TERMINAL_BREAK
                    #define GB_IF_TERMINAL_BREAK(z,zterminal)
                    #include "GB_reduce_to_scalar_template.c"
                }
                else
                { 
                    // break if terminal value reached
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 1
                    #undef  GB_TERMINAL_CONDITION
                    #define GB_TERMINAL_CONDITION(z,zterminal)  \
                            (memcmp (z, zterminal, zsize) == 0)
                    #undef  GB_IF_TERMINAL_BREAK
                    #define GB_IF_TERMINAL_BREAK(z,zterminal)   \
                            if (GB_TERMINAL_CONDITION (z, zterminal)) break
                    #include "GB_reduce_to_scalar_template.c"
                }

            }
            else
            {

                //--------------------------------------------------------------
                // generic worker: sum up the entries, with typecasting
                //--------------------------------------------------------------

                GB_BURBLE_MATRIX (A, "(generic reduce to scalar, with typecast:"
                    " %s) ", monoid->op->name) ;

                GB_cast_function
                    cast_A_to_Z = GB_cast_factory (ztype->code, A->type->code) ;

                // t += (ztype) Ax [p], with typecast
                #undef  GB_GETA_AND_UPDATE
                #define GB_GETA_AND_UPDATE(t,Ax,p)                      \
                    GB_void awork [GB_VLA(zsize)] ;                     \
                    cast_A_to_Z (awork, Ax +((p)*asize), asize) ;       \
                    freduce (t, t, awork)

                if (zterminal == NULL)
                { 
                    // monoid is not terminal
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 0
                    #undef  GB_TERMINAL_CONDITION
                    #define GB_TERMINAL_CONDITION(z,zterminal) 0
                    #undef  GB_IF_TERMINAL_BREAK
                    #define GB_IF_TERMINAL_BREAK
                    #include "GB_reduce_to_scalar_template.c"
                }
                else
                { 
                    // break if terminal value reached
                    #undef  GB_MONOID_IS_TERMINAL
                    #define GB_MONOID_IS_TERMINAL 1
                    #undef  GB_TERMINAL_CONDITION
                    #define GB_TERMINAL_CONDITION(z,zterminal)  \
                            (memcmp (z, zterminal, zsize) == 0)
                    #undef  GB_IF_TERMINAL_BREAK
                    #define GB_IF_TERMINAL_BREAK(z,zterminal)   \
                            if (GB_TERMINAL_CONDITION (z, zterminal)) break
                    #include "GB_reduce_to_scalar_template.c"
                }
            }
            info = GrB_SUCCESS ;
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // c = z or c = accum (c,z)
    //--------------------------------------------------------------------------

    // This operation does not use GB_accum_mask, since c and z are
    // scalars, not matrices.  There is no scalar mask.

    if (accum == NULL)
    { 
        // c = (ctype) z
        GB_cast_function
            cast_Z_to_C = GB_cast_factory (ctype->code, ztype->code) ;
        cast_Z_to_C (c, z, ctype->size) ;
    }
    else
    { 
        GxB_binary_function faccum = accum->binop_function ;

        GB_cast_function cast_C_to_xaccum, cast_Z_to_yaccum, cast_zaccum_to_C ;
        cast_C_to_xaccum = GB_cast_factory (accum->xtype->code, ctype->code) ;
        cast_Z_to_yaccum = GB_cast_factory (accum->ytype->code, ztype->code) ;
        cast_zaccum_to_C = GB_cast_factory (ctype->code, accum->ztype->code) ;

        // scalar workspace
        GB_void xaccum [GB_VLA(accum->xtype->size)] ;
        GB_void yaccum [GB_VLA(accum->ytype->size)] ;
        GB_void zaccum [GB_VLA(accum->ztype->size)] ;

        // xaccum = (accum->xtype) c
        cast_C_to_xaccum (xaccum, c, ctype->size) ;

        // yaccum = (accum->ytype) z
        cast_Z_to_yaccum (yaccum, z, zsize) ;

        // zaccum = xaccum "+" yaccum
        faccum (zaccum, xaccum, yaccum) ;

        // c = (ctype) zaccum
        cast_zaccum_to_C (c, zaccum, ctype->size) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

