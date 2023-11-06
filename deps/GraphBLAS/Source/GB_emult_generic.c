//------------------------------------------------------------------------------
// GB_emult_generic: generic methods for eWiseMult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_emult_generic handles the generic case for eWiseMult, when no built-in
// worker in the switch factory can handle this case.  This occurs for
// user-defined operators, when typecasting occurs, or for FIRST[IJ]* and
// SECOND[IJ]* positional operators.

// C is not iso, but A and/or B might be.

#include "GB_ewise.h"
#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_ek_slice.h"
#include "GB_generic.h"

GrB_Info GB_emult_generic       // generic emult
(
    // input/output:
    GrB_Matrix C,           // output matrix, static header
    // input:
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,  // array of structs
    const int C_ntasks,                         // # of tasks
    const int C_nthreads,                       // # of threads to use
    // analysis from phase0:
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const int C_sparsity,
    // from GB_emult_sparsity
    const int ewise_method,
    // from GB_emult_04, GB_emult_03, GB_emult_02:
    const int64_t *restrict Cp_kfirst,
    // to slice M, A, and/or B,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads,
    const int64_t *B_ek_slicing, const int B_ntasks, const int B_nthreads,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for ewise generic", GB0) ;
    ASSERT_MATRIX_OK (A, "A for ewise generic", GB0) ;
    ASSERT_MATRIX_OK (B, "B for ewise generic", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for ewise generic", GB0) ;
    ASSERT (!C->iso) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    const GrB_Type ctype = C->type ;
    const GB_Type_code ccode = ctype->code ;

    //--------------------------------------------------------------------------
    // get the opcode and define the typecasting functions
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;

    const bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    const bool op_is_first  = (opcode == GB_FIRST_binop_code) ;
    const bool op_is_second = (opcode == GB_SECOND_binop_code) ;
    const bool op_is_pair   = (opcode == GB_PAIR_binop_code) ;
    const bool A_is_pattern = (op_is_second || op_is_pair || op_is_positional) ;
    const bool B_is_pattern = (op_is_first  || op_is_pair || op_is_positional) ;

    const GxB_binary_function fop = op->binop_function ; // NULL if positional
    const size_t csize = ctype->size ;
    const size_t asize = A->type->size ;
    const size_t bsize = B->type->size ;
    const GrB_Type xtype = op->xtype ;
    const GrB_Type ytype = op->ytype ;

    const size_t xsize = (A_is_pattern) ? 1 : xtype->size ;
    const size_t ysize = (B_is_pattern) ? 1 : ytype->size ;
    const size_t zsize = op->ztype->size ;

    const GB_cast_function cast_A_to_X =
        (A_is_pattern) ? NULL : GB_cast_factory (xtype->code, A->type->code) ;

    const GB_cast_function cast_B_to_Y = 
        (B_is_pattern) ? NULL : GB_cast_factory (ytype->code, B->type->code) ;

    const GB_cast_function cast_Z_to_C =
        GB_cast_factory (ccode, op->ztype->code) ;
    
    // declare aij as xtype
    #define GB_DECLAREA(aij)                                            \
        GB_void aij [GB_VLA(xsize)] ;

    // aij = (xtype) A(i,j), located in Ax [pA]
    #define GB_GETA(aij,Ax,pA,A_iso)                                    \
        if (cast_A_to_X != NULL)                                        \
        {                                                               \
            cast_A_to_X (aij, Ax +((A_iso) ? 0:(pA)*asize), asize) ;    \
        }

    // declare bij as ytype
    #define GB_DECLAREB(bij)                                            \
        GB_void bij [GB_VLA(ysize)] ;

    // bij = (ytype) B(i,j), located in Bx [pB]
    #define GB_GETB(bij,Bx,pB,B_iso)                                    \
        if (cast_B_to_Y != NULL)                                        \
        {                                                               \
            cast_B_to_Y (bij, Bx +((B_iso) ? 0:(pB)*bsize), bsize) ;    \
        }

    #include "GB_ewise_shared_definitions.h"

    //--------------------------------------------------------------------------
    // do the ewise operation
    //--------------------------------------------------------------------------

    // C (i,j) = (ctype) z
    #undef  GB_PUTC
    #define GB_PUTC(z, Cx, p) cast_Z_to_C (Cx +((p)*csize), &z, csize)

    if (op_is_positional)
    {

        //----------------------------------------------------------------------
        // C(i,j) = positional_op (aij, bij)
        //----------------------------------------------------------------------

        bool depends_on_j ;
        const int64_t offset = GB_positional_offset (opcode, NULL,
            &depends_on_j) ;
        const bool index_is_i = 
            (opcode == GB_FIRSTI_binop_code  ) ||
            (opcode == GB_FIRSTI1_binop_code ) ||
            (opcode == GB_SECONDI_binop_code ) ||
            (opcode == GB_SECONDI1_binop_code) ;

        if (op->ztype == GrB_INT64)
        {

            // z = op (aij, bij)
            #undef  GB_BINOP
            #define GB_BINOP(z,x,y,i,j)                             \
                z = ((index_is_i)? (i):(j)) + offset

            // C(i,j) = positional_op (aij, bij)
            #undef  GB_EWISEOP
            #define GB_EWISEOP(Cx, p, aij, bij, i, j)               \
            {                                                       \
                int64_t z ;                                         \
                GB_BINOP (z, , , i, j) ;                            \
                GB_PUTC (z, Cx, p) ;                                \
            }

            if (ewise_method == GB_EMULT_METHOD2)
            { 
                // C=A.*B or C<#M>=A.*B; A sparse/hyper; M and B bitmap/full
                // C is sparse
                #include "GB_emult_02_template.c"
            }
            else if (ewise_method == GB_EMULT_METHOD3)
            { 
                // C=A.*B or C<#M>=A.*B; B sparse/hyper; M and A bitmap/full
                // C is sparse
                #include "GB_emult_03_template.c"
            }
            else if (ewise_method == GB_EMULT_METHOD4)
            { 
                // C<M>=A.*B; M sparse/hyper, A and B bitmap/full
                // C is sparse
                #include "GB_emult_04_template.c"
            }
            else if (C_sparsity == GxB_BITMAP)
            { 
                // C is bitmap: emult methods 5, 6, or 7
                #include "GB_emult_bitmap_template.c"
            }
            else
            { 
                // C is sparse: emult method 8 (abcdefgh)
                #include "GB_emult_08_meta.c"
            }
        }
        else
        {

            // C(i,j) = positional_op (aij, bij)
            #undef  GB_EWISEOP
            #define GB_EWISEOP(Cx, p, aij, bij, i, j)               \
            {                                                       \
                int64_t z ;                                         \
                GB_BINOP (z, , , i, j) ;                            \
                int32_t z32 = (int32_t) z ;                         \
                GB_PUTC (z32, Cx, p) ;                              \
            }

            if (ewise_method == GB_EMULT_METHOD2)
            { 
                // C=A.*B or C<#M>=A.*B; A sparse/hyper; M and B bitmap/full
                // C is sparse
                #include "GB_emult_02_template.c"
            }
            else if (ewise_method == GB_EMULT_METHOD3)
            { 
                // C=A.*B or C<#M>=A.*B; B sparse/hyper; M and A bitmap/full
                // C is sparse
                #include "GB_emult_03_template.c"
            }
            else if (ewise_method == GB_EMULT_METHOD4)
            { 
                // C<M>=A.*B; M sparse/hyper, A and B bitmap/full
                // C is sparse
                #include "GB_emult_04_template.c"
            }
            else if (C_sparsity == GxB_BITMAP)
            { 
                // C is bitmap: emult methods 5, 6, or 7
                #include "GB_emult_bitmap_template.c"
            }
            else
            { 
                // C is sparse: emult method 8 (abcdefgh)
                #include "GB_emult_08_meta.c"
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // standard binary operator
        //----------------------------------------------------------------------

        // z = op (aij, bij)
        #undef  GB_BINOP
        #define GB_BINOP(z, aij, bij, i, j)             \
            ASSERT (fop != NULL) ;                      \
            fop (z, aij, bij) ;

        // C(i,j) = (ctype) (A(i,j) + B(i,j))
        #undef  GB_EWISEOP
        #define GB_EWISEOP(Cx, p, aij, bij, i, j)       \
        {                                               \
            GB_void z [GB_VLA(zsize)] ;                 \
            GB_BINOP (z, aij, bij, i, j) ;              \
            GB_PUTC (z, Cx, p) ;                        \
        }

        // C(i,j) = (ctype) (A(i,j) + B(i,j))
        if (ewise_method == GB_EMULT_METHOD2)
        { 
            // emult method 2 (abc)
            // C=A.*B or C<#M>=A.*B; A sparse/hyper; M and B bitmap/full
            // C is sparse
            #include "GB_emult_02_template.c"
        }
        else if (ewise_method == GB_EMULT_METHOD3)
        { 
            // emult method 3 (abc)
            // C=A.*B or C<#M>=A.*B; B sparse/hyper; M and A bitmap/full
            // C is sparse
            #include "GB_emult_03_template.c"
        }
        else if (ewise_method == GB_EMULT_METHOD4)
        { 
            // C<M>=A.*B; M sparse/hyper, A and B bitmap/full
            // C is sparse
            #include "GB_emult_04_template.c"
        }
        else if (C_sparsity == GxB_BITMAP)
        { 
            // C is bitmap: emult methods 5, 6, or 7
            #include "GB_emult_bitmap_template.c"
        }
        else
        { 
            // C is sparse: emult method 8 (abcdefgh)
            #include "GB_emult_08_meta.c"
        }
    }

    ASSERT_MATRIX_OK (C, "C from ewise generic", GB0) ;
    return (GrB_SUCCESS) ;
}

