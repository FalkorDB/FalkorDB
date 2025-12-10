//------------------------------------------------------------------------------
// GB_AxB_saxpy5_lv.c: C+=A*B when C is full
//------------------------------------------------------------------------------

#if GB_Z_NBITS == 64
    #define VSETVL(x) __riscv_vsetvl_e64m8(x)
    #define VLE(x,y) __riscv_vle64_v_f64m8(x, y)
    #define VFMACC(x,y,z,w) __riscv_vfmacc_vf_f64m8(x, y, z, w)
    #define VSE(x,y,z) __riscv_vse64_v_f64m8(x, y, z)
    #define VECTORTYPE vfloat64m8_t
#else
    #define VSETVL(x) __riscv_vsetvl_e32m8(x)
    #define VLE(x,y) __riscv_vle32_v_f32m8(x, y)
    #define VFMACC(x,y,z,w) __riscv_vfmacc_vf_f32m8(x, y, z, w)
    #define VSE(x,y,z) __riscv_vse32_v_f32m8(x, y, z)
    #define VECTORTYPE vfloat32m8_t
#endif

{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int64_t m = C->vlen;     // # of rows of C and A
    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const bool B_iso = B->iso ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *)A->x;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *)B->x;
    // get the max number of elements that vector can store
    size_t vl = VSETVL(m);
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *)C->x;

    //--------------------------------------------------------------------------
    // C += A*B where A is full (and not iso or pattern-only)
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int tid = 0; tid < ntasks; tid++)
    {
        // get the task descriptor
        const int64_t jB_start = B_slice[tid];
        const int64_t jB_end = B_slice[tid + 1];
        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start; jB < jB_end; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBh_B (Bh, jB) ;
            GB_C_TYPE *restrict Cxj = Cx + (j * m) ;
            const int64_t pB_start = GB_IGET (Bp, jB) ;
            const int64_t pB_end   = GB_IGET (Bp, jB+1) ;

            //------------------------------------------------------------------
            // C(:,j) += A*B(:,j), on sets of vl rows of C and A at a time
            //------------------------------------------------------------------

            for (int64_t i = 0; i < m && (m - i) >= vl; i += vl)
            {
                // get C(i:i+vl,j)
                VECTORTYPE vc = VLE(Cxj + i, vl);
                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    // bkj = B(k,j)
                    const int64_t k = GB_IGET (Bi, pB) ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    // get A(i,k)
                    VECTORTYPE va = VLE(Ax + i + k * m, vl);
                    // C(i:i+15,j) += A(i:i+15,k)*B(k,j)
                    vc = VFMACC(vc, bkj, va, vl);
                }
                // save C(i:i+15,j)
                VSE(Cxj + i, vc, vl);
            }

            //------------------------------------------------------------------
            // lines 179-1036 from GB_AxB_saxpy5_unrolled.c
            //------------------------------------------------------------------

            int64_t remaining = m % vl;
            if (remaining > 0)
            {
                int64_t i = m - remaining;
                VECTORTYPE vc = VLE(Cxj + i, remaining);
                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    const int64_t k = GB_IGET (Bi, pB) ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    VECTORTYPE va = VLE(Ax + i + k * m, remaining);
                    vc = VFMACC(vc, bkj, va, remaining);
                }

                VSE(Cxj + i, vc, remaining);
            }
        }
    }
}
