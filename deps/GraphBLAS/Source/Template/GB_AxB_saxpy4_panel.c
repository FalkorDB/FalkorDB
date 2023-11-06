//------------------------------------------------------------------------------
// GB_AxB_saxpy4_panel.c: H += A*G or H = A*G for one panel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This methods handles both C bitmap (Template/GB_AxB_saxpy4_panel.c) and C
// full (Template/GB_AxB_saxbit_A_sparse_B_bitmap_template.c).  Those methods
// define the GB_HX_COMPUTE macro for use in this method, which computes the
// single update: H (i,jj) += A(i,k) * B(k,j).  The values of B have already
// been loaded into the panel G.

{
    switch (np)
    {

        case 4 : 

            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {
                // get A(:,k)
                const int64_t k = GBH_A (Ah, kA) ;
                // get B(k,j1:j2-1)
                #if GB_B_IS_BITMAP
                const int8_t gb0 = Gb [k          ] ;
                const int8_t gb1 = Gb [k +   bvlen] ;
                const int8_t gb2 = Gb [k + 2*bvlen] ;
                const int8_t gb3 = Gb [k + 3*bvlen] ;
                if (!(gb0 || gb1 || gb2 || gb3)) continue ;
                #endif
                GB_DECLAREB (gk0) ;
                GB_DECLAREB (gk1) ;
                GB_DECLAREB (gk2) ;
                GB_DECLAREB (gk3) ;
                GB_GETB (gk0, Gx, k          , B_iso) ;
                GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                GB_GETB (gk2, Gx, k + 2*bvlen, B_iso) ;
                GB_GETB (gk3, Gx, k + 3*bvlen, B_iso) ;
                // H += A(:,k)*B(k,j1:j2-1)
                const int64_t pA_end = Ap [kA+1] ;
                for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                { 
                    const int64_t i = Ai [pA] ;
                    const int64_t pH = i * 4 ;
                    GB_DECLAREA (aik) ;
                    GB_GETA (aik, Ax, pA, A_iso) ;
                    GB_HX_COMPUTE (gk0, gb0, 0) ;
                    GB_HX_COMPUTE (gk1, gb1, 1) ;
                    GB_HX_COMPUTE (gk2, gb2, 2) ;
                    GB_HX_COMPUTE (gk3, gb3, 3) ;
                }
            }
            break ;

        case 3 : 

            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {
                // get A(:,k)
                const int64_t k = GBH_A (Ah, kA) ;
                // get B(k,j1:j2-1)
                #if GB_B_IS_BITMAP
                const int8_t gb0 = Gb [k          ] ;
                const int8_t gb1 = Gb [k +   bvlen] ;
                const int8_t gb2 = Gb [k + 2*bvlen] ;
                if (!(gb0 || gb1 || gb2)) continue ;
                #endif
                GB_DECLAREB (gk0) ;
                GB_DECLAREB (gk1) ;
                GB_DECLAREB (gk2) ;
                GB_GETB (gk0, Gx, k          , B_iso) ;
                GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                GB_GETB (gk2, Gx, k + 2*bvlen, B_iso) ;
                // H += A(:,k)*B(k,j1:j2-1)
                const int64_t pA_end = Ap [kA+1] ;
                for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                { 
                    const int64_t i = Ai [pA] ;
                    const int64_t pH = i * 3 ;
                    GB_DECLAREA (aik) ;
                    GB_GETA (aik, Ax, pA, A_iso) ;
                    GB_HX_COMPUTE (gk0, gb0, 0) ;
                    GB_HX_COMPUTE (gk1, gb1, 1) ;
                    GB_HX_COMPUTE (gk2, gb2, 2) ;
                }
            }
            break ;

        case 2 : 

            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {
                // get A(:,k)
                const int64_t k = GBH_A (Ah, kA) ;
                // get B(k,j1:j2-1)
                #if GB_B_IS_BITMAP
                const int8_t gb0 = Gb [k          ] ;
                const int8_t gb1 = Gb [k +   bvlen] ;
                if (!(gb0 || gb1)) continue ;
                #endif
                // H += A(:,k)*B(k,j1:j2-1)
                GB_DECLAREB (gk0) ;
                GB_DECLAREB (gk1) ;
                GB_GETB (gk0, Gx, k          , B_iso) ;
                GB_GETB (gk1, Gx, k +   bvlen, B_iso) ;
                const int64_t pA_end = Ap [kA+1] ;
                for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                { 
                    const int64_t i = Ai [pA] ;
                    const int64_t pH = i * 2 ;
                    GB_DECLAREA (aik) ;
                    GB_GETA (aik, Ax, pA, A_iso) ;
                    GB_HX_COMPUTE (gk0, gb0, 0) ;
                    GB_HX_COMPUTE (gk1, gb1, 1) ;
                }
            }
            break ;

        case 1 : 

            for (int64_t kA = 0 ; kA < anvec ; kA++)
            {
                // get A(:,k)
                const int64_t k = GBH_A (Ah, kA) ;
                // get B(k,j1:j2-1) where j1 == j2-1
                #if GB_B_IS_BITMAP
                const int8_t gb0 = Gb [k] ;
                if (!gb0) continue ;
                #endif
                // H += A(:,k)*B(k,j1:j2-1)
                GB_DECLAREB (gk0) ;
                GB_GETB (gk0, Gx, k, B_iso) ;
                const int64_t pA_end = Ap [kA+1] ;
                for (int64_t pA = Ap [kA] ; pA < pA_end ; pA++)
                { 
                    const int64_t i = Ai [pA] ;
                    const int64_t pH = i ;
                    GB_DECLAREA (aik) ;
                    GB_GETA (aik, Ax, pA, A_iso) ;
                    GB_HX_COMPUTE (gk0, 1, 0) ;
                }
            }
            break ;

        default:;
    }

    #undef GB_HX_COMPUTE
    #undef GB_B_kj_PRESENT
    #undef GB_MULT_A_ik_G_kj
}

