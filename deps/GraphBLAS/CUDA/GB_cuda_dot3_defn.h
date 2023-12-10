//------------------------------------------------------------------------------
// CUDA/GB_cuda_dot3_defn.h: definitions just for the dot3 method
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#pragma once

//------------------------------------------------------------------------------
// operators
//------------------------------------------------------------------------------

#if GB_C_ISO

//  GB_MULTADD now defined in header
//  #define GB_MULTADD( c, a ,b, i, k, j)
    #define GB_DOT_TERMINAL( c ) break
    #define GB_DOT_MERGE(pA,pB)                                         \
    {                                                                   \
        cij_exists = true ;                                             \
    }
    #define GB_CIJ_EXIST_POSTCHECK

#else

//  GB_MULTADD now defined in header
//  #define GB_MULTADD( c, a, b, i, k, j )                              \
//  {                                                                   \
//      GB_Z_TYPE x_op_y ;                                              \
//      GB_MULT ( x_op_y, a, b, i, k, j ) ; /* x_op_y = a*b */          \
//      GB_ADD ( c, c, x_op_y ) ;           /* c += x_op_y  */          \
//  }

    #define GB_DOT_TERMINAL( c ) GB_IF_TERMINAL_BREAK ( c, zterminal )

    #if GB_IS_PLUS_PAIR_REAL_SEMIRING

        // cij += A(k,i) * B(k,j), for merge operation (plus_pair_real semiring)
        #if GB_Z_IGNORE_OVERFLOW
            // plus_pair for int64, uint64, float, or double
            #define GB_DOT_MERGE(pA,pB) cij++ ;
            #define GB_CIJ_EXIST_POSTCHECK cij_exists = (cij != 0) ;
        #else
            // plus_pair semiring for small integers
            #define GB_DOT_MERGE(pA,pB)                                     \
            {                                                               \
                cij_exists = true ;                                         \
                cij++ ;                                                     \
            }
            #define GB_CIJ_EXIST_POSTCHECK
        #endif

    #else

        // cij += A(k,i) * B(k,j), for merge operation (general case)
        #define GB_DOT_MERGE(pA,pB)                                         \
        {                                                                   \
            GB_GETA ( aki, Ax, pA, ) ;      /* aki = A(k,i) */              \
            GB_GETB ( bkj, Bx, pB, ) ;      /* bkj = B(k,j) */              \
            cij_exists = true ;                                             \
            GB_MULTADD ( cij, aki, bkj, i, k, j ) ;  /* cij += aki * bkj */ \
        }
        #define GB_CIJ_EXIST_POSTCHECK

    #endif

#endif

