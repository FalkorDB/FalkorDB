//------------------------------------------------------------------------------
// GB_apply_op: typecast and apply a unary/binary/idxunop operator to an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Cx = op (A)

// Cx and A->x may be aliased.

// This function is CSR/CSC agnostic.  For positional ops, A is treated as if
// it is in CSC format.  The caller has already modified the op if A is in CSR
// format.

#include "apply/GB_apply.h"
#include "binaryop/GB_binop.h"
#include "slice/GB_ek_slice.h"
#include "include/GB_unused.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_control.h"
#include "FactoryKernels/GB_uop__include.h"
#include "FactoryKernels/GB_ew__include.h"
#endif

#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

GrB_Info GB_apply_op        // apply a unary op, idxunop, or binop, Cx = op (A)
(
    GB_void *Cx,                    // output array
    const GrB_Type ctype,           // type of C
    const GB_iso_code C_code_iso,   // C non-iso, or code to compute C iso value
        const GB_Operator op_in,    // unary/index-unary/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, C=binop(s,A), else C=binop(A,s)
        bool flipij,                // if true, flip i,j for user idxunop
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_Operator op = op_in ;
    ASSERT (Cx != NULL) ;
    ASSERT_MATRIX_OK (A, "A input for GB_apply_op", GB0) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // A can be jumbled
    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    ASSERT (GB_IMPLIES (op != NULL, ctype == op->ztype)) ;
    ASSERT_SCALAR_OK_OR_NULL (scalar, "scalar for GB_apply_op", GB0) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    // A->x is not const since the operator might be applied in-place, if
    // C is aliased to C.

    GB_void *Ax = (GB_void *) A->x ;        // A->x has type A->type
    const int8_t *Ab = A->b ;               // only if A is bitmap
    const GrB_Type Atype = A->type ;        // type of A->x
    const int64_t anz = GB_nnz_held (A) ;   // size of A->x and Cx

    //--------------------------------------------------------------------------
    // determine the maximum number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    GB_Opcode opcode ;
    bool op_is_unop = false ;
    bool op_is_binop = false ;
    bool is64 = false ;
    bool is32 = false ;

    if (op != NULL)
    { 
        opcode = op->opcode ;
        op_is_unop = GB_IS_UNARYOP_CODE (opcode) ;
        op_is_binop = GB_IS_BINARYOP_CODE (opcode) ;
        ASSERT (!GB_IS_INDEXBINARYOP_CODE (opcode)) ;
        is64 = (op->ztype == GrB_INT64) ;
        is32 = (op->ztype == GrB_INT32) ;

        if (GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (opcode))
        {
            // rename builtin positional binary ops to positional unary ops
            GrB_UnaryOp op1 = NULL ;
            switch (opcode)
            {
                case GB_FIRSTI_binop_code    : // z = first_i(A(i,j),y) == i
                case GB_SECONDI_binop_code   : // z = second_i(x,A(i,j)) == i
                    // rename FIRSTI and SECONDI to POSITIONI
                    op1 = is64 ? GxB_POSITIONI_INT64 : GxB_POSITIONI_INT32 ;
                    break ;

                case GB_FIRSTI1_binop_code   : // z = first_i1(A(i,j),y) == i+1
                case GB_SECONDI1_binop_code  : // z = second_i1(x,A(i,j)) == i+1
                    // rename FIRSTI1 and SECONDI1 to POSITIONI1
                    op1 = is64 ? GxB_POSITIONI1_INT64 : GxB_POSITIONI1_INT32 ;
                    break ;

                case GB_FIRSTJ_binop_code    : // z = first_j(A(i,j),y) == j
                case GB_SECONDJ_binop_code   : // z = second_j(x,A(i,j)) == j
                    // rename FIRSTJ and SECONDJ to POSITIONJ
                    op1 = is64 ? GxB_POSITIONJ_INT64 : GxB_POSITIONJ_INT32 ;
                    break ;

                case GB_FIRSTJ1_binop_code   : // z = first_j1(A(i,j),y) == j+1
                case GB_SECONDJ1_binop_code  : // z = second_j1(x,A(i,j)) == j+1
                    // rename FIRSTJ1 and SECONDJ1 to POSITIONJ1
                    op1 = is64 ? GxB_POSITIONJ1_INT64 : GxB_POSITIONJ1_INT32 ;
                    break ;

                default:;
            }
            ASSERT (op1 != NULL) ;
            op = (GB_Operator) op1 ;
            opcode = op->opcode ;
            op_is_unop = true ;
            op_is_binop = false ;
            ASSERT (GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode)) ;
        }

    }
    else
    { 
        // C is iso, with no operator to apply; just call GB_unop_iso below.
        ASSERT (C_code_iso == GB_ISO_1 ||   // C iso value is 1
                C_code_iso == GB_ISO_S ||   // C iso value is the scalar
                C_code_iso == GB_ISO_A) ;   // C iso value is the iso value of A
        opcode = GB_NOP_code ;
    }

    //--------------------------------------------------------------------------
    // determine number of threads to use and slice the A matrix if needed
    //--------------------------------------------------------------------------

    int64_t anvec = A->nvec ;
    int A_ntasks = 0 ;
    int A_nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    info = GrB_NO_VALUE ;
    bool depends_on_j = (opcode == GB_USER_idxunop_code) ;
    int64_t thunk = 0 ;

    if (GB_OPCODE_IS_POSITIONAL (opcode))
    { 
        thunk = GB_positional_offset (opcode, scalar, &depends_on_j) ;
    }

    if (depends_on_j)
    { 
        // slice the entries for each task
        GB_SLICE_MATRIX (A, 32) ;
    }

    //--------------------------------------------------------------------------
    // apply the operator
    //--------------------------------------------------------------------------

    if (GB_OPCODE_IS_POSITIONAL (opcode))
    {

        //----------------------------------------------------------------------
        // via the positional kernel
        //----------------------------------------------------------------------

        ASSERT_OP_OK (op, "positional unop/idxunop: GB_apply_op", GB0) ;

        #if defined ( GRAPHBLAS_HAS_CUDA )
        if (GB_cuda_apply_unop_branch (ctype, A, op)) {
            info = GB_cuda_apply_unop (Cx, ctype, op, flipij, A,
                (GB_void *) &thunk) ;
        } 
        #endif
        
        if (info == GrB_NO_VALUE)
        {
            // get A and C
            const int64_t *restrict Ah = A->h ;
            const int64_t *restrict Ap = A->p ;
            const int64_t *restrict Ai = A->i ;
            int64_t avlen = A->vlen ;

            //------------------------------------------------------------------
            // Cx = positional_op (A)
            //------------------------------------------------------------------

            if (is64)
            { 

                //--------------------------------------------------------------
                // int64 Cx = positional_op (A)
                //--------------------------------------------------------------

                int64_t *restrict Cz = (int64_t *) Cx ;
                switch (opcode)
                {

                    case GB_POSITIONI_unop_code  : // z = pos_i(A(i,j)) == i
                    case GB_POSITIONI1_unop_code : // z = pos_i1(A(i,j)) == i+1
                    case GB_ROWINDEX_idxunop_code : // z = i+thunk
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (i + thunk) ;
                        #include "apply/template/GB_apply_unop_ip.c"
                        break ;

                    case GB_POSITIONJ_unop_code  : // z = pos_j(A(i,j)) == j
                    case GB_POSITIONJ1_unop_code : // z = pos_j1(A(i,j)) == j+1
                    case GB_COLINDEX_idxunop_code : // z = j+thunk
                        #define GB_APPLY_OP(pC,pA)                  \
                            Cz [pC] = (j + thunk) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    case GB_DIAGINDEX_idxunop_code : // z = (j-(i+thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (j - (i+thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    case GB_FLIPDIAGINDEX_idxunop_code : // z = (i-(j+thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (i - (j+thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    default: ;
                }

            }
            else if (is32)
            { 

                //--------------------------------------------------------------
                // int32 Cx = positional_op (A)
                //--------------------------------------------------------------

                int32_t *restrict Cz = (int32_t *) Cx ;
                switch (opcode)
                {

                    case GB_POSITIONI_unop_code  : // z = pos_i(A(i,j)) == i
                    case GB_POSITIONI1_unop_code : // z = pos_i1(A(i,j)) == i+1
                    case GB_ROWINDEX_idxunop_code : // z = i+thunk
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (int32_t) (i + thunk) ;
                        #include "apply/template/GB_apply_unop_ip.c"
                        break ;

                    case GB_POSITIONJ_unop_code  : // z = pos_j(A(i,j)) == j
                    case GB_POSITIONJ1_unop_code : // z = pos_j1(A(i,j)) == j+1
                    case GB_COLINDEX_idxunop_code : // z = j+thunk
                        #define GB_APPLY_OP(pC,pA)                  \
                            Cz [pC] = (int32_t) (j + thunk) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    case GB_DIAGINDEX_idxunop_code : // z = (j-(i+thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (int32_t) (j - (i+thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    case GB_FLIPDIAGINDEX_idxunop_code : // z = (i-(j+thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (int32_t) (i - (j+thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ;

                    default: ;
                }

            }
            else
            { 

                //--------------------------------------------------------------
                // bool Cx = positional_op (A)
                //--------------------------------------------------------------

                ASSERT (op->ztype == GrB_BOOL) ;
                bool *restrict Cz = (bool *) Cx ;
                switch (opcode)
                {

                    case GB_TRIL_idxunop_code : // z = (j <= (i+thunk))
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (j <= (i + thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_TRIU_idxunop_code : // z = (j >= (i+thunk))
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (j >= (i + thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_DIAG_idxunop_code : // z = (j == (i+thunk))
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (j == (i + thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_OFFDIAG_idxunop_code : // z = (j != (i+thunk))
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (j != (i + thunk)) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_COLLE_idxunop_code : // z = (j <= thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            Cz [pC] = (j <= thunk) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_COLGT_idxunop_code : // z = (j > thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            Cz [pC] = (j > thunk) ;
                        #include "apply/template/GB_apply_unop_ijp.c"
                        break ; ;

                    case GB_ROWLE_idxunop_code : // z = (i <= thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (i <= thunk) ;
                        #include "apply/template/GB_apply_unop_ip.c"
                        break ; ;

                    case GB_ROWGT_idxunop_code : // z = (i > thunk)
                        #define GB_APPLY_OP(pC,pA)                  \
                            int64_t i = GBI (Ai, pA, avlen) ;       \
                            Cz [pC] = (i > thunk) ;
                        #include "apply/template/GB_apply_unop_ip.c"
                        break ; ;

                    default: ;
                }
            }
            info = GrB_SUCCESS ;
        }
    }
    else if (C_code_iso != GB_NON_ISO)
    {

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        // if C is iso, this function takes O(1) time
        GBURBLE ("(iso apply) ") ;
        ASSERT_MATRIX_OK (A, "A passing to GB_unop_iso", GB0) ;
        if (anz > 0)
        { 
            // Cx [0] = unop (A), binop (scalar,A), or binop (A,scalar)
            GB_unop_iso (Cx, ctype, C_code_iso, op, A, scalar) ;
        }

        info = GrB_SUCCESS ;

    }
    else if (op_is_unop)
    {

        //----------------------------------------------------------------------
        // unop via the factory kernel
        //----------------------------------------------------------------------

        ASSERT_OP_OK (op, "unop for GB_apply_op", GB0) ;
        ASSERT (!A->iso) ;

        #if defined ( GRAPHBLAS_HAS_CUDA )
        if (GB_cuda_apply_unop_branch (ctype, A, op)) {
            info = GB_cuda_apply_unop (Cx, ctype, op, flipij, A, NULL) ;
        } 
        #endif

        // determine number of threads to use
        #ifndef GBCOMPACT
        if (info == GrB_NO_VALUE)
        {
            GB_IF_FACTORY_KERNELS_ENABLED
            { 
                if (Atype == op->xtype || opcode == GB_IDENTITY_unop_code)
                { 

                    // The switch factory is used if the op is IDENTITY, or if
                    // no typecasting.  IDENTITY operator can do arbitrary
                    // typecasting (it is not used if no typecasting is done).

                    //----------------------------------------------------------
                    // define the worker for the switch factory
                    //----------------------------------------------------------

                    #define GB_uop_apply(unop,zname,aname) \
                        GB (_uop_apply_ ## unop ## zname ## aname)

                    #define GB_WORKER(unop,zname,ztype,aname,atype)          \
                    {                                                        \
                        info = GB_uop_apply (unop,zname,aname) (Cx, Ax, Ab,  \
                            anz, A_nthreads) ;                               \
                    }                                                        \
                    break ;

                    //----------------------------------------------------------
                    // launch the switch factory
                    //----------------------------------------------------------

                    #include "apply/factory/GB_unop_factory.c"
                }
            }
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_apply_unop_jit (Cx, ctype, op, flipij, A,
                NULL, NULL, 0, A_nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            GB_BURBLE_N (anz, "(generic unop apply: %s) ", op->name) ;

            size_t asize = Atype->size ;
            size_t zsize = op->ztype->size ;
            size_t xsize = op->xtype->size ;
            GB_Type_code acode = Atype->code ;
            GB_Type_code xcode = op->xtype->code ;
            GB_cast_function cast_A_to_X = GB_cast_factory (xcode, acode) ;
            GxB_unary_function fop = op->unop_function ;

            #define GB_APPLY_OP(pC,pA)                          \
                /* xwork = (xtype) Ax [pA] */                   \
                GB_void xwork [GB_VLA(xsize)] ;                 \
                cast_A_to_X (xwork, Ax +(pA)*asize, asize) ;    \
                /* Cx [pC] = fop (xwork) */                     \
                fop (Cx +((pC)*zsize), xwork) ;

            #include "apply/template/GB_apply_unop_ip.c"

            info = GrB_SUCCESS ;
        }

    }
    else if (op_is_binop)
    { 

        //----------------------------------------------------------------------
        // apply a binary operator (bound to a scalar)
        //----------------------------------------------------------------------

        ASSERT_OP_OK (op, "standard binop for GB_apply_op", GB0) ;
        ASSERT_SCALAR_OK (scalar, "scalar for GB_apply_op", GB0) ;

        GB_Type_code xcode, ycode, zcode ;
        ASSERT (opcode != GB_FIRST_binop_code) ;
        ASSERT (opcode != GB_SECOND_binop_code) ;
        ASSERT (opcode != GB_PAIR_binop_code) ;
        ASSERT (opcode != GB_ANY_binop_code) ;
        ASSERT (opcode != GB_USER_idxbinop_code) ;

        size_t asize = Atype->size ;
        size_t ssize = scalar->type->size ;
        size_t zsize = op->ztype->size ;
        size_t xsize = op->xtype->size ;
        size_t ysize = op->ytype->size ;

        GB_Type_code scalar_code = scalar->type->code ;
        xcode = op->xtype->code ;
        ycode = op->ytype->code ;

        // typecast the scalar to the operator input
        size_t ssize_cast ;
        GB_Type_code scalar_code_cast ;
        if (binop_bind1st)
        { 
            ssize_cast = xsize ;
            scalar_code_cast = xcode ;
        }
        else
        { 
            ssize_cast = ysize ;
            scalar_code_cast = ycode ;
        }
        GB_void swork [GB_VLA(ssize_cast)] ;
        GB_void *scalarx = (GB_void *) scalar->x ;
        if (scalar_code_cast != scalar_code)
        { 
            // typecast the scalar to the operator input, in swork
            GB_cast_function cast_s =
                GB_cast_factory (scalar_code_cast, scalar_code) ;
            cast_s (swork, scalar->x, ssize) ;
            scalarx = swork ;
        }

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        if (binop_bind1st)
        {

            //------------------------------------------------------------------
            // z = binop (scalar,Ax)
            //------------------------------------------------------------------

            #if defined ( GRAPHBLAS_HAS_CUDA )
            if (GB_cuda_apply_binop_branch (ctype, (GrB_BinaryOp) op, A))
            {
                info = GB_cuda_apply_binop (Cx, ctype, (GrB_BinaryOp) op, A,
                    scalarx, true) ;
            } 
            #endif

            #ifndef GBCOMPACT
            if (info == GrB_NO_VALUE)
            {
                GB_IF_FACTORY_KERNELS_ENABLED
                { 
                    if (GB_binop_builtin (op->xtype, false, Atype, false,
                        (GrB_BinaryOp) op, false, &opcode, &xcode, &ycode,
                        &zcode))
                    { 

                        //------------------------------------------------------
                        // define the worker for the switch factory
                        //------------------------------------------------------

                        #define GB_bind1st(binop,xname) \
                            GB (_bind1st_ ## binop ## xname)
                        #define GB_BINOP_WORKER(binop,xname)                   \
                        {                                                      \
                            info = GB_bind1st (binop, xname) (Cx, scalarx, Ax, \
                                Ab, anz, A_nthreads) ;                         \
                        }                                                      \
                        break ;

                        //------------------------------------------------------
                        // launch the switch factory
                        //------------------------------------------------------

                        #define GB_NO_FIRST
                        #define GB_NO_SECOND
                        #define GB_NO_PAIR
                        #include "binaryop/factory/GB_binop_factory.c"
                    }
                }
            }
            #endif

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                info = GB_apply_bind1st_jit (Cx, ctype,
                    (GrB_BinaryOp) op, scalarx, A, A_nthreads) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // z = binop (Ax,scalar)
            //------------------------------------------------------------------
            
            #if defined ( GRAPHBLAS_HAS_CUDA )
            if (GB_cuda_apply_binop_branch (ctype, (GrB_BinaryOp) op, A))
            {
                info = GB_cuda_apply_binop (Cx, ctype, (GrB_BinaryOp) op, A,
                scalarx, false) ;
            } 
            #endif


            #ifndef GBCOMPACT
            if (info == GrB_NO_VALUE)
            {
                GB_IF_FACTORY_KERNELS_ENABLED
                {  
                    if (GB_binop_builtin (Atype, false, op->ytype, false,
                        (GrB_BinaryOp) op, false, &opcode, &xcode, &ycode,
                        &zcode))
                    { 

                        //------------------------------------------------------
                        // define the worker for the switch factory
                        //------------------------------------------------------

                        #define GB_bind2nd(binop,xname) \
                            GB (_bind2nd_ ## binop ## xname)
                        #undef  GB_BINOP_WORKER
                        #define GB_BINOP_WORKER(binop,xname)                  \
                        {                                                     \
                            info = GB_bind2nd (binop, xname) (Cx, Ax, scalarx,\
                                Ab, anz, A_nthreads) ;                        \
                        }                                                     \
                        break ;

                        //------------------------------------------------------
                        // launch the switch factory
                        //------------------------------------------------------

                        #define GB_NO_FIRST
                        #define GB_NO_SECOND
                        #define GB_NO_PAIR
                        #include "binaryop/factory/GB_binop_factory.c"
                    }
                }
            }
            #endif

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                info = GB_apply_bind2nd_jit (Cx, ctype,
                    (GrB_BinaryOp) op, A, scalarx, A_nthreads) ;
            }
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        {

            GB_BURBLE_N (anz, "(generic binop apply: %s) ", op->name) ;

            GB_Type_code acode = Atype->code ;
            GxB_binary_function fop = op->binop_function ;
            ASSERT (fop != NULL) ;
            ASSERT (!A->iso) ;

            if (binop_bind1st)
            { 
                // Cx = binop (scalar,Ax) with bind1st
                GB_cast_function cast_A_to_Y = GB_cast_factory (ycode, acode) ;
                #define GB_APPLY_OP(pC,pA)                          \
                    /* ywork = (ytype) Ax [pA] */                   \
                    GB_void ywork [GB_VLA(ysize)] ;                 \
                    cast_A_to_Y (ywork, Ax +(pA)*asize, asize) ;    \
                    /* Cx [pC] = fop (scalarx, ywork) */            \
                    fop (Cx +((pC)*zsize), scalarx, ywork) ;
                #include "apply/template/GB_apply_unop_ip.c"
            }
            else
            { 
                // Cx = binop (Ax,scalar) with bind2nd
                GB_cast_function cast_A_to_X = GB_cast_factory (xcode, acode) ;
                #define GB_APPLY_OP(pC,pA)                          \
                    /* xwork = (xtype) Ax [pA] */                   \
                    GB_void xwork [GB_VLA(xsize)] ;                 \
                    cast_A_to_X (xwork, Ax +(pA)*asize, asize) ;    \
                    /* Cx [pC] = fop (xwork, scalarx) */            \
                    fop (Cx +((pC)*zsize), xwork, scalarx) ;
                #include "apply/template/GB_apply_unop_ip.c"
            }
            info = GrB_SUCCESS ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // apply a user-defined index_unary op
        //----------------------------------------------------------------------

        // All valued GrB_IndexUnaryOps (GrB_VALUE*) have already been
        // renamed to their corresponding binary op (GrB_VALUEEQ_FP32
        // became GrB_EQ_FP32, for example).  The only remaining index
        // unary ops are positional, and user-defined.  Positional ops have
        // been handled above, so only user-defined index unary ops are
        // left.

        ASSERT (opcode == GB_USER_idxunop_code) ;
        size_t ssize = scalar->type->size ;
        size_t ysize = op->ytype->size ;

        GB_Type_code scalar_code = scalar->type->code ;
        GB_Type_code ycode = op->ytype->code ;

        GB_void ywork [GB_VLA(ysize)] ;
        GB_void *ythunk = (GB_void *) scalar->x ;
        if (ycode != scalar_code)
        { 
            // typecast the scalar to the operator input, in ywork
            GB_cast_function cast_s = GB_cast_factory (ycode, scalar_code) ;
            cast_s (ywork, scalar->x, ssize) ;
            ythunk = ywork ;
        }

        #if defined ( GRAPHBLAS_HAS_CUDA )
        if (GB_cuda_apply_unop_branch (ctype, A, op)) {
            info = GB_cuda_apply_unop (Cx, ctype, op, flipij, A, ythunk) ;
        } 
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_apply_unop_jit (Cx, ctype, op, flipij, A,
                ythunk, A_ek_slicing, A_ntasks, A_nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            GB_BURBLE_N (anz, "(generic apply: user-defined idxunop) ") ;

            // get A and C
            const int64_t *restrict Ah = A->h ;
            const int64_t *restrict Ap = A->p ;
            const int64_t *restrict Ai = A->i ;
            int64_t avlen = A->vlen ;
            GB_Type_code acode = Atype->code ;

            // A can be iso-valued, but C is not
            bool A_iso = A->iso ;
            ASSERT (C_code_iso == GB_NON_ISO) ;
            if (A_iso)
            { 
                GB_BURBLE_N (anz, "(A iso; C non-iso) ") ;
            }

            GxB_index_unary_function fop = op->idxunop_function ;
            size_t asize = Atype->size ;
            size_t zsize = op->ztype->size ;
            size_t xsize = op->xtype->size ;
            GB_Type_code xcode = op->xtype->code ;
            GB_cast_function cast_A_to_X = GB_cast_factory (xcode, acode) ;

            // Cx [pC] = op (Ax [A_iso ? 0 : pA], i, j, ythunk)
            #define GB_APPLY_OP(pC,pA)                                      \
                int64_t i = GBI (Ai, pA, avlen) ;                           \
                GB_void xwork [GB_VLA(xsize)] ;                             \
                cast_A_to_X (xwork, Ax +(A_iso ? 0 : (pA))*asize, asize) ;  \
                fop (Cx +((pC)*zsize), xwork,                               \
                    flipij ? j : i, flipij ? i : j, ythunk) ;
            #include "apply/template/GB_apply_unop_ijp.c"
            info = GrB_SUCCESS ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (info) ;
}

