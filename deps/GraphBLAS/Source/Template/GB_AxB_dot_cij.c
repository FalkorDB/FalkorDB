//------------------------------------------------------------------------------
// GB_AxB_dot_cij: compute C(i,j) = A(:,i)'*B(:,j)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Computes C(i,j) = A (:,i)'*B(:,j) via sparse dot product.  This template is
// used for all three cases: C=A'*B, C<M>=A'*B, and C<!M>=A'*B in dot2 when C
// is bitmap, and for C<M>=A'*B when C and M are sparse or hyper in dot3.

// if A_NOT_TRANSPOSED is #defined, then dot2 is computing A(i,:)*B(:,j)
// for C<#M>=A*B.  In this case A is either bitmap or full, and B is always
// sparse.

// When used as the multiplicative operator, the PAIR operator provides some
// useful special cases.  Its output is always one, for any matching pair of
// entries A(k,i)'*B(k,j) for some k.  If the monoid is ANY, then C(i,j)=1 if
// the intersection for the dot product is non-empty.  This intersection has to
// be found, in general.  However, suppose B(:,j) is dense.  Then every entry
// in the pattern of A(:,i)' will produce a 1 from the PAIR operator.  If the
// monoid is ANY, then C(i,j)=1 if A(:,i)' is nonempty.  If the monoid is PLUS,
// then C(i,j) is simply nnz(A(:,i)), assuming no overflow.  The XOR monoid
// acts like a 1-bit summation, so the result of the XOR_PAIR_BOOL semiring
// will be C(i,j) = mod (nnz(A(:,i)'*B(:,j)),2).

// If both A(:,i) and B(:,j) are sparse, then the intersection must still be
// found, so these optimizations can be used only if A(:,i) and/or B(:,j) are
// entirely populated.

// The #include'ing file must use GB_DECLARE_TERMINAL_CONST (zterminal),
// or define zterminal another way (see Template/GB_AxB_dot_generic.c).

#ifndef GB_MXM_SHARED_DEFINITIONS_H
#error "undefined"
#endif

#undef GB_A_INDEX
#ifdef GB_A_NOT_TRANSPOSED
#define GB_A_INDEX(k) (pA+(k)*vlen)
#else
#define GB_A_INDEX(k) (pA+(k))
#endif

//------------------------------------------------------------------------------
// C(i,j) = A(:,i)'*B(:,j): a single dot product
//------------------------------------------------------------------------------

{
    int64_t pB = pB_start ;
    ASSERT (vlen > 0) ;
    #if ( GB_B_IS_SPARSE || GB_B_IS_HYPER )
    ASSERT (bjnz > 0) ;
    #endif
    #if ( GB_A_IS_SPARSE || GB_A_IS_HYPER )
    ASSERT (ainz > 0) ;
    #endif

    #if ( GB_A_IS_FULL && GB_B_IS_FULL )
    {

        //----------------------------------------------------------------------
        // both A and B are full
        //----------------------------------------------------------------------

        #if GB_IS_ANY_PAIR_SEMIRING
        { 
            // nothing to do; C is iso
        }
        #elif GB_IS_LXOR_PAIR_SEMIRING
        { 
            // (boolean XOR monoid)_PAIR semiring
            cij = (GB_C_TYPE) (((uint64_t) vlen) & 0x1L) ;
        }
        #elif GB_IS_PLUS_PAIR_8_SEMIRING
        { 
            // (PLUS int8, uint8 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) vlen) & 0xFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_16_SEMIRING
        { 
            // (PLUS int16, uint16 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) vlen) & 0xFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_32_SEMIRING
        { 
            // (PLUS int32, uint32 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) vlen) & 0xFFFFFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_BIG_SEMIRING
        { 
            // (PLUS int64, uint64, float, or double)_PAIR semirings
            cij = (GB_C_TYPE) vlen ;
        }
//      #elif GB_IS_PLUS_PAIR_FC32_SEMIRING
//      {
//          // (PLUS monoid for float complex)_PAIR semiring
//          cij = GJ_CMPLX32 ((float) vlen, 0) ;
//      }
//      #elif GB_IS_PLUS_PAIR_FC64_SEMIRING
//      {
//          // (PLUS monoid for double complex)_PAIR semiring
//          cij = GJ_CMPLX64 ((double) vlen, 0) ;
//      }
        #elif GB_IS_MIN_FIRSTJ_SEMIRING
        { 
            // MIN_FIRSTJ semiring: take the first entry
            cij = GB_OFFSET ;
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        { 
            // MAX_FIRSTJ semiring: take the last entry
            cij = vlen-1 + GB_OFFSET ;
        }
        #else
        {
            // cij = A(0,i) * B(0,j)
            GB_DECLAREA (aki) ;
            GB_GETA (aki, Ax, pA, A_iso) ;      // aki = A(0,i)
            GB_DECLAREB (bkj) ;
            GB_GETB (bkj, Bx, pB, B_iso) ;      // bkj = B(0,j)
            GB_MULT (cij, aki, bkj, i, 0, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
            for (int64_t k = 1 ; k < vlen ; k++)
            { 
                // break if cij terminal
                GB_IF_TERMINAL_BREAK (cij, zterminal) ;
                // cij += A(k,i) * B(k,j)
                GB_GETA (aki, Ax, pA+k, A_iso) ;    // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k, B_iso) ;    // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ; // cij += aki * bkj
            }
        }
        #endif
        GB_DOT_ALWAYS_SAVE_CIJ ;

    }
    #elif ( GB_A_IS_FULL && GB_B_IS_BITMAP )
    {

        //----------------------------------------------------------------------
        // A is full and B is bitmap
        //----------------------------------------------------------------------

        #if GB_IS_MIN_FIRSTJ_SEMIRING
        {
            // MIN_FIRSTJ semiring: take the first entry in B(:,j)
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        {
            // MAX_FIRSTJ semiring: take the last entry in B(:,j)
            for (int64_t k = vlen-1 ; k >= 0 ; k--)
            {
                if (Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #else
        {
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Bb [pB+k])
                { 
                    GB_DOT (k, pA+k, pB+k) ;
                }
            }
        }
        #endif
        GB_DOT_SAVE_CIJ ;

    }
    #elif ( GB_A_IS_FULL && ( GB_B_IS_SPARSE || GB_B_IS_HYPER ) )
    {

        //----------------------------------------------------------------------
        // A is full and B is sparse/hyper (C = A'*B or A*B)
        //----------------------------------------------------------------------

        #if GB_IS_ANY_PAIR_SEMIRING
        { 
            // nothing to do; C is iso
        }
        #elif GB_IS_LXOR_PAIR_SEMIRING
        { 
            // (boolean XOR monoid)_PAIR semiring
            cij = (GB_C_TYPE) (((uint64_t) bjnz) & 0x1L) ;
        }
        #elif GB_IS_PLUS_PAIR_8_SEMIRING
        { 
            // (PLUS int8, uint8 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) bjnz) & 0xFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_16_SEMIRING
        { 
            // (PLUS int16, uint16 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) bjnz) & 0xFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_32_SEMIRING
        { 
            // (PLUS int32, uint32 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) bjnz) & 0xFFFFFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_BIG_SEMIRING
        { 
            // (PLUS int64, uint64, float, or double)_PAIR semirings
            cij = (GB_C_TYPE) bjnz ;
        }
//      #elif GB_IS_PLUS_PAIR_FC32_SEMIRING
//      {
//          // (PLUS monoid for float complex)_PAIR semiring
//          cij = GJ_CMPLX32 ((float) bjnz, 0) ;
//      }
//      #elif GB_IS_PLUS_PAIR_FC64_SEMIRING
//      {
//          // (PLUS monoid for double complex)_PAIR semiring
//          cij = GJ_CMPLX64 ((double) bjnz, 0) ;
//      }
        #elif GB_IS_MIN_FIRSTJ_SEMIRING
        { 
            // MIN_FIRSTJ semiring: take the first entry in B(:,j)
            cij = Bi [pB] + GB_OFFSET ;
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        { 
            // MAX_FIRSTJ semiring: take the last entry in B(:,j)
            cij = Bi [pB_end-1] + GB_OFFSET ;
        }
        #else
        {
            int64_t k = Bi [pB] ;               // first row index of B(:,j)
            // cij = (A(k,i) or A(i,k)) * B(k,j)
            GB_DECLAREA (aki) ;
            GB_GETA (aki, Ax, GB_A_INDEX(k), A_iso) ; // aki = A(k,i) or A(i,k)
            GB_DECLAREB (bkj) ;
            GB_GETB (bkj, Bx, pB  , B_iso) ;    // bkj = B(k,j)
            GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
            for (int64_t p = pB+1 ; p < pB_end ; p++)
            { 
                // break if cij terminal
                GB_IF_TERMINAL_BREAK (cij, zterminal) ;
                int64_t k = Bi [p] ;
                // cij += (A(k,i) or A(i,k)) * B(k,j)
                GB_GETA (aki, Ax, GB_A_INDEX(k), A_iso) ; //aki=A(k,i) or A(i,k)
                GB_GETB (bkj, Bx, p, B_iso) ;           // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
            }
        }
        #endif
        GB_DOT_ALWAYS_SAVE_CIJ ;

    }
    #elif ( GB_A_IS_BITMAP && GB_B_IS_FULL )
    {

        //----------------------------------------------------------------------
        // A is bitmap and B is full
        //----------------------------------------------------------------------

        #if GB_IS_MIN_FIRSTJ_SEMIRING
        {
            // MIN_FIRSTJ semiring: take the first entry in A(:,i)
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Ab [pA+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        {
            // MAX_FIRSTJ semiring: take the last entry in A(:,i)
            for (int64_t k = vlen-1 ; k >= 0 ; k--)
            {
                if (Ab [pA+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #else
        {
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Ab [pA+k])
                { 
                    GB_DOT (k, pA+k, pB+k) ;
                }
            }
        }
        #endif
        GB_DOT_SAVE_CIJ ;

    }
    #elif ( GB_A_IS_BITMAP && GB_B_IS_BITMAP )
    {

        //----------------------------------------------------------------------
        // both A and B are bitmap
        //----------------------------------------------------------------------

        #if GB_IS_MIN_FIRSTJ_SEMIRING
        {
            // MIN_FIRSTJ semiring: take the first entry
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Ab [pA+k] && Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        {
            // MAX_FIRSTJ semiring: take the last entry
            for (int64_t k = vlen-1 ; k >= 0 ; k--)
            {
                if (Ab [pA+k] && Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #else
        {
            for (int64_t k = 0 ; k < vlen ; k++)
            {
                if (Ab [pA+k] && Bb [pB+k])
                { 
                    GB_DOT (k, pA+k, pB+k) ;
                }
            }
        }
        #endif
        GB_DOT_SAVE_CIJ ;

    }
    #elif ( GB_A_IS_BITMAP && ( GB_B_IS_SPARSE || GB_B_IS_HYPER ) )
    {

        //----------------------------------------------------------------------
        // A is bitmap and B is sparse/hyper (C = A'*B or A*B)
        //----------------------------------------------------------------------

        #if GB_IS_MIN_FIRSTJ_SEMIRING
        {
            // MIN_FIRSTJ semiring: take the first entry
            for (int64_t p = pB ; p < pB_end ; p++)
            {
                int64_t k = Bi [p] ;
                if (Ab [GB_A_INDEX (k)])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        {
            // MAX_FIRSTJ semiring: take the last entry
            for (int64_t p = pB_end-1 ; p >= pB ; p--)
            {
                int64_t k = Bi [p] ;
                if (Ab [GB_A_INDEX (k)])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #else
        {
            for (int64_t p = pB ; p < pB_end ; p++)
            {
                int64_t k = Bi [p] ;
                if (Ab [GB_A_INDEX (k)])
                { 
                    GB_DOT (k, GB_A_INDEX (k), p) ;
                }
            }
        }
        #endif
        GB_DOT_SAVE_CIJ ;

    }
    #elif ( (GB_A_IS_SPARSE || GB_A_IS_HYPER) && GB_B_IS_FULL )
    {

        //----------------------------------------------------------------------
        // A is sparse/hyper and B is full
        //----------------------------------------------------------------------

        #if GB_IS_ANY_PAIR_SEMIRING
        { 
            // nothing to do; C is iso
        }
        #elif GB_IS_LXOR_PAIR_SEMIRING
        { 
            // (boolean XOR monoid)_PAIR semiring
            cij = (GB_C_TYPE) (((uint64_t) ainz) & 0x1L) ;
        }
        #elif GB_IS_PLUS_PAIR_8_SEMIRING
        { 
            // (PLUS int8, uint8 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) ainz) & 0xFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_16_SEMIRING
        { 
            // (PLUS int16, uint16 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) ainz) & 0xFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_32_SEMIRING
        { 
            // (PLUS int32, uint32 monoids)_PAIR semirings
            cij = (GB_C_TYPE) (((uint64_t) ainz) & 0xFFFFFFFFL) ;
        }
        #elif GB_IS_PLUS_PAIR_BIG_SEMIRING
        { 
            // (PLUS int64, uint64, float, or double)_PAIR semirings
            cij = (GB_C_TYPE) ainz ;
        }
//      #elif GB_IS_PLUS_PAIR_FC32_SEMIRING
//      {
//          // (PLUS monoid for float complex)_PAIR semiring
//          cij = GJ_CMPLX32 ((float) ainz, 0) ;
//      }
//      #elif GB_IS_PLUS_PAIR_FC64_SEMIRING
//      {
//          // (PLUS monoid for double complex)_PAIR semiring
//          cij = GJ_CMPLX64 ((double) ainz, 0) ;
//      }
        #elif GB_IS_MIN_FIRSTJ_SEMIRING
        { 
            // MIN_FIRSTJ semiring: take the first entry in A(:,i)
            cij = Ai [pA] + GB_OFFSET ;
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        { 
            // MAX_FIRSTJ semiring: take the last entry in A(:,i)
            cij = Ai [pA_end-1] + GB_OFFSET ;
        }
        #else
        {
            int64_t k = Ai [pA] ;               // first row index of A(:,i)
            // cij = A(k,i) * B(k,j)
            GB_DECLAREA (aki) ;
            GB_GETA (aki, Ax, pA  , A_iso) ;    // aki = A(k,i)
            GB_DECLAREB (bkj) ;
            GB_GETB (bkj, Bx, pB+k, B_iso) ;    // bkj = B(k,j)
            GB_MULT (cij, aki, bkj, i, k, j) ;  // cij = aki * bkj
            GB_PRAGMA_SIMD_REDUCTION_MONOID (cij)
            for (int64_t p = pA+1 ; p < pA_end ; p++)
            { 
                // break if cij terminal
                GB_IF_TERMINAL_BREAK (cij, zterminal) ;
                int64_t k = Ai [p] ;
                // cij += A(k,i) * B(k,j)
                GB_GETA (aki, Ax, p   , A_iso) ;    // aki = A(k,i)
                GB_GETB (bkj, Bx, pB+k, B_iso) ;    // bkj = B(k,j)
                GB_MULTADD (cij, aki, bkj, i, k, j) ;   // cij += aki * bkj
            }
        }
        #endif
        GB_DOT_ALWAYS_SAVE_CIJ ;

    }
    #elif ( (GB_A_IS_SPARSE || GB_A_IS_HYPER) && GB_B_IS_BITMAP )
    {

        //----------------------------------------------------------------------
        // A is sparse/hyper and B is bitmap
        //----------------------------------------------------------------------

        #if GB_IS_MIN_FIRSTJ_SEMIRING
        {
            // MIN_FIRSTJ semiring: take the first entry
            for (int64_t p = pA ; p < pA_end ; p++)
            {
                int64_t k = Ai [p] ;
                if (Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #elif GB_IS_MAX_FIRSTJ_SEMIRING
        {
            // MAX_FIRSTJ semiring: take the last entry
            for (int64_t p = pA_end-1 ; p >= pA ; p--)
            {
                int64_t k = Ai [p] ;
                if (Bb [pB+k])
                { 
                    cij_exists = true ;
                    cij = k + GB_OFFSET ;
                    break ;
                }
            }
        }
        #else
        {
            for (int64_t p = pA ; p < pA_end ; p++)
            {
                int64_t k = Ai [p] ;
                if (Bb [pB+k])
                { 
                    GB_DOT (k, p, pB+k) ;
                }
            }
        }
        #endif
        GB_DOT_SAVE_CIJ ;

    }
    #else
    {

        //----------------------------------------------------------------------
        // both A and B are sparse/hyper
        //----------------------------------------------------------------------

        // The MIN_FIRSTJ semirings are exploited, by terminating as soon as
        // any entry is found.  The MAX_FIRSTJ semirings are not treated
        // specially here.  They could be done with a backwards traversal of
        // the sparse vectors A(:,i) and B(:,j).

        if (Ai [pA_end-1] < ib_first || ib_last < Ai [pA])
        { 

            //------------------------------------------------------------------
            // pattern of A(:,i) and B(:,j) don't overlap; C(i,j) doesn't exist
            //------------------------------------------------------------------

            ASSERT (!GB_CIJ_EXISTS) ;

        }
        else if (ainz > 8 * bjnz)
        {

            //------------------------------------------------------------------
            // B(:,j) is very sparse compared to A(:,i)
            //------------------------------------------------------------------

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    // discard all entries A(ia:ib-1,i)
                    int64_t pleft = pA + 1 ;
                    int64_t pright = pA_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ib, Ai, pleft, pright) ;
                    ASSERT (pleft > pA) ;
                    pA = pleft ;
                }
                else if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    pB++ ;
                }
                else // ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT (ia, pA, pB) ;
                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    break ;
                    #endif
                    pA++ ;
                    pB++ ;
                }
            }
            GB_DOT_SAVE_CIJ ;

        }
        else if (bjnz > 8 * ainz)
        {

            //------------------------------------------------------------------
            // A(:,i) is very sparse compared to B(:,j)
            //------------------------------------------------------------------

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    pA++ ;
                }
                else if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    // discard all entries B(ib:ia-1,j)
                    int64_t pleft = pB + 1 ;
                    int64_t pright = pB_end - 1 ;
                    GB_TRIM_BINARY_SEARCH (ia, Bi, pleft, pright) ;
                    ASSERT (pleft > pB) ;
                    pB = pleft ;
                }
                else // ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT (ia, pA, pB) ;
                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    break ;
                    #endif
                    pA++ ;
                    pB++ ;
                }
            }
            GB_DOT_SAVE_CIJ ;

        }
        else
        {

            //------------------------------------------------------------------
            // A(:,i) and B(:,j) have about the same sparsity
            //------------------------------------------------------------------

            while (pA < pA_end && pB < pB_end)
            {
                int64_t ia = Ai [pA] ;
                int64_t ib = Bi [pB] ;
                if (ia < ib)
                { 
                    // A(ia,i) appears before B(ib,j)
                    pA++ ;
                }
                else if (ib < ia)
                { 
                    // B(ib,j) appears before A(ia,i)
                    pB++ ;
                }
                else // ia == ib == k
                { 
                    // A(k,i) and B(k,j) are the next entries to merge
                    GB_DOT (ia, pA, pB) ;
                    #if GB_IS_MIN_FIRSTJ_SEMIRING
                    break ;
                    #endif
                    pA++ ;
                    pB++ ;
                }
            }
            GB_DOT_SAVE_CIJ ;
        }
    }
    #endif
}

