//------------------------------------------------------------------------------
// GB_AxB_saxpy_generic: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy_generic computes C=A*B, C<M>=A*B, or C<!M>=A*B in parallel,
// with arbitrary types and operators, via memcpy and function pointers.  C can
// be hyper, sparse, or bitmap (never full).  For all cases, the four matrices
// C, M (if present), A, and B have the same format (by-row or by-column), or
// they represent implicitly transposed matrices with the same effect.  This
// method does not handle the dot-product methods, which compute C=A'*B if A
// and B are held by column, or equivalently A*B' if both are held by row.

// This method uses GB_AxB_saxpy3_generic_* and GB_AxB_saxbit_generic_*
// to implement two meta-methods, each of which can contain further specialized
// methods (such as the fine/ coarse x Gustavson/Hash, mask/no-mask methods in
// saxpy3):

// saxpy3: general purpose method, where C is sparse or hypersparse,
//          via GB_AxB_saxpy3_template.c.  SaxpyTasks holds the (fine/coarse x
//          Gustavson/Hash) tasks constructed by GB_AxB_saxpy3_slice*.

// saxbit: general purpose method, where C is bitmap, via
//          GB_AxB_saxbit_template.c.

// C is not iso, and it is never full.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_binop.h"
#include "GB_AxB_saxpy_generic.h"

GrB_Info GB_AxB_saxpy_generic
(
    GrB_Matrix C,                   // any sparsity
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,          // ignored if C is bitmap
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const int saxpy_method,         // saxpy3 or bitmap method
    const int ntasks,
    const int nthreads,
    // for saxpy3 only:
    GB_saxpy3task_struct *restrict SaxpyTasks, // NULL if C is bitmap
    const int nfine,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk,
    // for saxbit only:
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_slice,
    const int64_t *restrict H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
)
{

    //--------------------------------------------------------------------------
    // get operators, functions, workspace, contents of A, B, and C
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    GrB_BinaryOp mult = semiring->multiply ;
    GB_Opcode opcode = mult->opcode ;

    //--------------------------------------------------------------------------
    // C = A*B via saxpy3 or bitmap method, function pointers, and typecasting
    //--------------------------------------------------------------------------

    if (GB_OPCODE_IS_POSITIONAL (opcode))
    { 

        //----------------------------------------------------------------------
        // generic semirings with positional mulitiply operators
        //----------------------------------------------------------------------

        GB_BURBLE_MATRIX (C, "(generic positional C=A*B) ") ;

        ASSERT (!flipxy) ;

        // C always has type int64_t or int32_t.  The monoid must be used via
        // its function pointer.  The positional multiply operator must be
        // hard-coded since it has no function pointer.  The numerical values
        // and types of A and B are not accessed.

        if (mult->ztype == GrB_INT64)
        {
            switch (opcode)
            {
                case GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
                case GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_firsti64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_firsti64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;

                case GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
                case GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
                case GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
                case GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j))== k+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_firstj64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_firstj64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;

                case GB_SECONDJ_binop_code  :   // z = second_j(x,B(k,j)) == j
                case GB_SECONDJ1_binop_code :   // z = second_j1(x,B(k,j))== j+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_secondj64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_secondj64 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;
                default: ;
            }
        }
        else
        {
            switch (opcode)
            {
                case GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
                case GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_firsti32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_firsti32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;

                case GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
                case GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
                case GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
                case GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j))== k+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_firstj32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_firstj32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;

                case GB_SECONDJ_binop_code  :   // z = second_j(x,B(k,j)) == j
                case GB_SECONDJ1_binop_code :   // z = second_j1(x,B(k,j))== j+1
                    {
                        if (saxpy_method == GB_SAXPY_METHOD_3)
                        { 
                            // C is sparse or hypersparse
                            info = GB_AxB_saxpy3_generic_secondj32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              SaxpyTasks, nfine, do_sort, Werk) ;
                        }
                        else
                        { 
                            // C is bitmap
                            info = GB_AxB_saxbit_generic_secondj32 
                             (C, M, Mask_comp, Mask_struct, M_in_place,
                              A, A_is_pattern, B, B_is_pattern, semiring,
                              ntasks, nthreads,
                              nfine_tasks_per_vector, use_coarse_tasks,
                              use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                              A_slice, H_slice, Wcx, Wf) ;
                        }
                    }
                    break ;
                default: ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // generic semirings with standard multiply operators
        //----------------------------------------------------------------------

        GB_BURBLE_MATRIX (C, "(generic C=A*B) ") ;

        if (opcode == GB_FIRST_binop_code)
        {
            // t = A(i,k)
            // fmult is not used and can be NULL.  This is required for
            // GB_reduce_to_vector for user-defined types.
            ASSERT (!flipxy) ;
            ASSERT (B_is_pattern) ;
            if (saxpy_method == GB_SAXPY_METHOD_3)
            { 
                // C is sparse or hypersparse
                info = GB_AxB_saxpy3_generic_first 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      SaxpyTasks, nfine, do_sort, Werk) ;
            }
            else
            { 
                // C is bitmap
                info = GB_AxB_saxbit_generic_first 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      nfine_tasks_per_vector, use_coarse_tasks,
                      use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                      A_slice, H_slice, Wcx, Wf) ;
            }
        }
        else if (opcode == GB_SECOND_binop_code)
        {
            // t = B(i,k)
            // fmult is not used and can be NULL.  This is required for
            // GB_reduce_to_vector for user-defined types.
            ASSERT (!flipxy) ;
            ASSERT (A_is_pattern) ;
            if (saxpy_method == GB_SAXPY_METHOD_3)
            { 
                // C is sparse or hypersparse
                info = GB_AxB_saxpy3_generic_second 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      SaxpyTasks, nfine, do_sort, Werk) ;
            }
            else
            { 
                // C is bitmap
                info = GB_AxB_saxbit_generic_second 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      nfine_tasks_per_vector, use_coarse_tasks,
                      use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                      A_slice, H_slice, Wcx, Wf) ;
            }
        }
        else if (flipxy)
        {
            // t = B(k,j) * A(i,k)
            if (saxpy_method == GB_SAXPY_METHOD_3)
            { 
                // C is sparse or hypersparse, mult is flipped
                info = GB_AxB_saxpy3_generic_flipped 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      SaxpyTasks, nfine, do_sort, Werk) ;
            }
            else
            { 
                // C is bitmap, mult is flipped
                info = GB_AxB_saxbit_generic_flipped 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      nfine_tasks_per_vector, use_coarse_tasks,
                      use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                      A_slice, H_slice, Wcx, Wf) ;
            }
        }
        else
        {
            // t = A(i,k) * B(k,j)
            if (saxpy_method == GB_SAXPY_METHOD_3)
            { 
                // C is sparse or hypersparse, mult is unflipped
                info = GB_AxB_saxpy3_generic_unflipped 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      SaxpyTasks, nfine, do_sort, Werk) ;
            }
            else
            { 
                // C is bitmap, mult is unflipped
                info = GB_AxB_saxbit_generic_unflipped 
                     (C, M, Mask_comp, Mask_struct, M_in_place,
                      A, A_is_pattern, B, B_is_pattern, semiring,
                      ntasks, nthreads,
                      nfine_tasks_per_vector, use_coarse_tasks,
                      use_atomics, M_ek_slicing, M_nthreads, M_ntasks,
                      A_slice, H_slice, Wcx, Wf) ;
            }
        }
    }

    return (info) ;
}

