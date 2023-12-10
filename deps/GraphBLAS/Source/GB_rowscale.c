//------------------------------------------------------------------------------
// GB_rowscale: C = D*B, row scale with diagonal matrix D
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#include "GB_mxm.h"
#include "GB_binop.h"
#include "GB_apply.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_ew__include.h"
#endif

#define GB_FREE_ALL GB_phybix_free (C) ;

GrB_Info GB_rowscale                // C = D*B, row scale with diagonal D
(
    GrB_Matrix C,                   // output matrix, static header
    const GrB_Matrix D,             // diagonal input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=D*A
                                    // the monoid is not used
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_MATRIX_OK (D, "D for rowscale D*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for rowscale D*B", GB0) ;
    ASSERT (!GB_ZOMBIES (D)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (!GB_PENDING (D)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for numeric D*B", GB0) ;
    ASSERT (D->vdim == B->vlen) ;
    ASSERT (GB_is_diagonal (D)) ;

    ASSERT (!GB_IS_BITMAP (D)) ;        // bitmap or full: not needed
    ASSERT (!GB_IS_BITMAP (B)) ;
    ASSERT (!GB_IS_FULL (D)) ;

    GBURBLE ("(%s=%s*%s) ",
        GB_sparsity_char_matrix (B),    // C has the sparsity structure of B
        GB_sparsity_char_matrix (D),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Type ztype = mult->ztype ;
    ASSERT (ztype == semiring->add->op->ztype) ;
    GB_Opcode opcode = mult->opcode ;
    // GB_reduce_to_vector does not use GB_rowscale:
    ASSERT (!(mult->binop_function == NULL &&
        (opcode == GB_FIRST_binop_code || opcode == GB_SECOND_binop_code))) ;

    //--------------------------------------------------------------------------
    // determine if C is iso (ignore the monoid since it isn't used)
    //--------------------------------------------------------------------------

    size_t zsize = ztype->size ;
    GB_void cscalar [GB_VLA(zsize)] ;
    bool C_iso = GB_AxB_iso (cscalar, D, B, D->vdim, semiring, flipxy, true) ;

    //--------------------------------------------------------------------------
    // copy the pattern of B into C
    //--------------------------------------------------------------------------

    // allocate C->x but do not initialize it
    // set C->iso = C_iso   OK
    GB_OK (GB_dup_worker (&C, C_iso, B, false, ztype)) ;
    info = GrB_NO_VALUE ;

    //--------------------------------------------------------------------------
    // C = D*B, row scale, compute numerical values
    //--------------------------------------------------------------------------

    if (GB_OPCODE_IS_POSITIONAL (opcode))
    { 

        //----------------------------------------------------------------------
        // apply a positional operator: convert C=D*B to C=op(B)
        //----------------------------------------------------------------------

        // determine unary operator to compute C=D*B
        ASSERT (!flipxy) ;
        GrB_UnaryOp op = NULL ;
        if (ztype == GrB_INT64)
        {
            switch (opcode)
            {
                // first_op(D,B) becomes position_i(B)
                case GB_FIRSTI_binop_code   : 
                case GB_FIRSTJ_binop_code   : op = GxB_POSITIONI_INT64 ; break;
                case GB_FIRSTI1_binop_code  : 
                case GB_FIRSTJ1_binop_code  : op = GxB_POSITIONI1_INT64; break;
                // second_op(D,B) becomes position_op(B)
                case GB_SECONDI_binop_code  : op = GxB_POSITIONI_INT64 ; break;
                case GB_SECONDJ_binop_code  : op = GxB_POSITIONJ_INT64 ; break;
                case GB_SECONDI1_binop_code : op = GxB_POSITIONI1_INT64; break;
                case GB_SECONDJ1_binop_code : op = GxB_POSITIONJ1_INT64; break;
                default:  ;
            }
        }
        else
        {
            switch (opcode)
            {
                // first_op(D,B) becomes position_i(B)
                case GB_FIRSTI_binop_code   : 
                case GB_FIRSTJ_binop_code   : op = GxB_POSITIONI_INT32 ; break;
                case GB_FIRSTI1_binop_code  : 
                case GB_FIRSTJ1_binop_code  : op = GxB_POSITIONI1_INT32; break;
                // second_op(D,B) becomes position_op(B)
                case GB_SECONDI_binop_code  : op = GxB_POSITIONI_INT32 ; break;
                case GB_SECONDJ_binop_code  : op = GxB_POSITIONJ_INT32 ; break;
                case GB_SECONDI1_binop_code : op = GxB_POSITIONI1_INT32; break;
                case GB_SECONDJ1_binop_code : op = GxB_POSITIONJ1_INT32; break;
                default:  ;
            }
        }
        GB_OK (GB_apply_op (C->x, C->type, GB_NON_ISO,
            (GB_Operator) op,   // positional op
            NULL, false, false, B, Werk)) ;
        ASSERT_MATRIX_OK (C, "rowscale positional: C = D*B output", GB0) ;
        info = GrB_SUCCESS ;

    }
    else if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // C is iso; pattern already computed above
        GBURBLE ("(iso rowscale) ") ;
        memcpy (C->x, cscalar, zsize) ;
        info = GrB_SUCCESS ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C is non iso
        //----------------------------------------------------------------------

        //----------------------------------------------------------------------
        // determine if the values are accessed
        //----------------------------------------------------------------------

        bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
        bool op_is_second = (opcode == GB_SECOND_binop_code) ;
        bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;
        bool D_is_pattern = false ;
        bool B_is_pattern = false ;
        ASSERT (!op_is_pair) ;

        if (flipxy)
        { 
            // z = fmult (b,a) will be computed
            D_is_pattern = op_is_first  || op_is_pair ;
            B_is_pattern = op_is_second || op_is_pair ;
            ASSERT (GB_IMPLIES (!D_is_pattern,
                GB_Type_compatible (D->type, mult->ytype))) ;
            ASSERT (GB_IMPLIES (!B_is_pattern,
                GB_Type_compatible (B->type, mult->xtype))) ;
        }
        else
        { 
            // z = fmult (a,b) will be computed
            D_is_pattern = op_is_second || op_is_pair ;
            B_is_pattern = op_is_first  || op_is_pair ;
            ASSERT (GB_IMPLIES (!D_is_pattern,
                GB_Type_compatible (D->type, mult->xtype))) ;
            ASSERT (GB_IMPLIES (!B_is_pattern,
                GB_Type_compatible (B->type, mult->ytype))) ;
        }

        //----------------------------------------------------------------------
        // determine the number of threads to use
        //----------------------------------------------------------------------

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (GB_nnz_held (B) + B->nvec, chunk,
            nthreads_max) ;

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        info = GrB_NO_VALUE ;
        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_DxB(mult,xname) GB (_DxB_ ## mult ## xname)

            #define GB_BINOP_WORKER(mult,xname)                     \
            {                                                       \
                info = GB_DxB(mult,xname) (C, D, B, nthreads) ;     \
            }                                                       \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            GB_Type_code xcode, ycode, zcode ;
            if (GB_binop_builtin (D->type, D_is_pattern, B->type, B_is_pattern,
                mult, flipxy, &opcode, &xcode, &ycode, &zcode))
            { 
                // C=D*B, rowscale with built-in operator
                #define GB_BINOP_IS_SEMIRING_MULTIPLIER
                #define GB_NO_PAIR
                #include "GB_binop_factory.c"
                #undef  GB_BINOP_IS_SEMIRING_MULTIPLIER
            }
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_rowscale_jit (C, D, B, mult, flipxy, nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        {

            //------------------------------------------------------------------
            // C = D*B, row scale, with typecasting or user-defined operator
            //------------------------------------------------------------------

            //------------------------------------------------------------------
            // get operators, functions, workspace, contents of D, B, and C
            //------------------------------------------------------------------

            #include "GB_generic.h"
            GB_BURBLE_MATRIX (C, "(generic C=D*B rowscale) ") ;

            GxB_binary_function fmult = mult->binop_function ;

            size_t csize = C->type->size ;
            size_t dsize = D_is_pattern ? 0 : D->type->size ;
            size_t bsize = B_is_pattern ? 0 : B->type->size ;

            size_t xsize = mult->xtype->size ;
            size_t ysize = mult->ytype->size ;

            // scalar workspace: because of typecasting, the x/y types need not
            // be the same as the size of the D and B types.
            // flipxy false: dii = (xtype) D(i,i) and bij = (ytype) B(i,j)
            // flipxy true:  dii = (ytype) D(i,i) and bij = (xtype) B(i,j)
            size_t dii_size = flipxy ? ysize : xsize ;
            size_t bij_size = flipxy ? xsize : ysize ;

            GB_cast_function cast_D, cast_B ;
            if (flipxy)
            { 
                // D is typecasted to y, and B is typecasted to x
                cast_D = D_is_pattern ? NULL :
                         GB_cast_factory (mult->ytype->code, D->type->code) ;
                cast_B = B_is_pattern ? NULL :
                         GB_cast_factory (mult->xtype->code, B->type->code) ;
            }
            else
            { 
                // D is typecasted to x, and B is typecasted to y
                cast_D = D_is_pattern ? NULL :
                         GB_cast_factory (mult->xtype->code, D->type->code) ;
                cast_B = B_is_pattern ? NULL :
                         GB_cast_factory (mult->ytype->code, B->type->code) ;
            }

            //------------------------------------------------------------------
            // C = D*B via function pointers, and typecasting
            //------------------------------------------------------------------

            // dii = D(i,i), located in Dx [i]
            #define GB_DECLAREA(dii)                                    \
                GB_void dii [GB_VLA(dii_size)] ;
            #define GB_GETA(dii,Dx,i,D_iso)                             \
                if (!D_is_pattern)                                      \
                {                                                       \
                    cast_D (dii, Dx +(D_iso ? 0:(i)*dsize), dsize) ;    \
                }

            // bij = B(i,j), located in Bx [pB]
            #define GB_DECLAREB(bij)                                    \
                GB_void bij [GB_VLA(bij_size)] ;
            #define GB_GETB(bij,Bx,pB,B_iso)                            \
                if (!B_is_pattern)                                      \
                {                                                       \
                    cast_B (bij, Bx +(B_iso ? 0:(pB)*bsize), bsize) ;   \
                }

            #define GB_C_TYPE GB_void

            #include "GB_ewise_shared_definitions.h"

            if (flipxy)
            { 
                #undef  GB_EWISEOP
                #define GB_EWISEOP(Cx,p,x,y,i,j) fmult (Cx +((p)*csize),y,x)
                #include "GB_rowscale_template.c"
            }
            else
            { 
                #undef  GB_EWISEOP
                #define GB_EWISEOP(Cx,p,x,y,i,j) fmult (Cx +((p)*csize),x,y)
                #include "GB_rowscale_template.c"
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
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "rowscale: C = D*B output", GB0) ;
    return (GrB_SUCCESS) ;
}

