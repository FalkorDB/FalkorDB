#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "GpuTimer.h"
#include "GB_cuda_buckets.h"
#include "../../rmm_wrap/rmm_wrap.h"
#include <gtest/gtest.h>
#include "test_data.hpp"
extern "C" {
#include "GB.h"
}

#include "../GB_cuda_common_jitFactory.hpp"
#include "../GB_cuda_mxm_dot3_jitFactory.hpp"
#include "../GB_cuda_reduce_jitFactory.hpp"
#include "dataFactory.hpp"

template<typename T_C, typename T_M, typename T_A, typename T_B>
class mxm_problem_spec {

public:
    mxm_problem_spec(GrB_Monoid monoid_, GrB_BinaryOp binop_, int64_t N_, int64_t Annz_, int64_t Bnnz_, int64_t Cnnz_,
                     int sparsity_control_A_ = GxB_SPARSE, int sparsity_control_B_ = GxB_SPARSE) :
        mysemiring(), binop(binop_), monoid(monoid_), N(N_),
        G(N_, N_), Annz(Annz_), Bnnz(Bnnz_), Cnnz(Cnnz_), mask_struct(true), flipxy(false), mask_comp(false) {

        // FIXME: This should be getting set automatically somehow.
        float Cnzpercent = (float) Cnnz_/(N_*N_);

        // TODO: Allocate and fill arrays for buckets and nano buckets
        G.init_A(Annz_, sparsity_control_A_, GxB_BY_ROW);
        G.init_B(Bnnz_, sparsity_control_B_, GxB_BY_ROW);
        G.init_C(Cnzpercent);
//      G.fill_buckets( TB ); // all elements go to testbucket= TB

        /************************
         * Create mxm factory
         */
        auto grb_info = GrB_Semiring_new(&mysemiring, monoid_, binop_);
        GRB_TRY (grb_info) ;
        GrB_Matrix A = G.getA();
        GrB_Matrix B = G.getB();
        //GRB_TRY (GxB_Matrix_fprint (A, "A", GxB_SHORT_VERBOSE, stdout)) ;
        //GRB_TRY (GxB_Matrix_fprint (B, "B", GxB_SHORT_VERBOSE, stdout)) ;
    }

    ~mxm_problem_spec() {

        std::cout << "Calling G.del()" << std::endl;
        G.del();

    }

    GrB_Matrix getC(){ return G.getC(); }
    GrB_Matrix getM(){ return G.getM(); }
    GrB_Matrix getA(){ return G.getA(); }
    GrB_Matrix getB(){ return G.getB(); }

    GrB_Monoid getMonoid() { return monoid; }
    GrB_BinaryOp getBinaryOp() { return binop; }

    int64_t getN() { return N; }
    int64_t getAnnz() { return Annz; }
    int64_t getBnnz() { return Bnnz; }
    int64_t getCnnz() { return Cnnz; }

    auto &getG() { return G; }

    GB_cuda_mxm_factory &get_mxm_factory() {

        // Lazily create the mxm factory
        if(!mymxmfactory.has_value()) {

            mymxmfactory.emplace(GB_cuda_mxm_factory());
            GrB_Matrix C = G.getC();
            GrB_Matrix M = G.getM();
            GrB_Matrix A = G.getA();
            GrB_Matrix B = G.getB();

            bool C_iso = false ;
            int C_sparsity = GB_sparsity (M) ;
            GrB_Type ctype = binop->ztype ;

            (*mymxmfactory).mxm_factory (
                    C_iso, C_sparsity, ctype,
                    M, mask_struct, mask_comp,
                    mysemiring, flipxy,
                    A, B) ;
        }
        return *mymxmfactory;
    }
    GrB_Semiring get_semiring() { return mysemiring; }

    void set_sparsity_control(GrB_Matrix mat, int gxb_sparsity_control, int gxb_format) {
        GRB_TRY (GxB_Matrix_Option_set (mat, GxB_SPARSITY_CONTROL, gxb_sparsity_control)) ;
        GRB_TRY (GxB_Matrix_Option_set(mat, GxB_FORMAT, gxb_format));
        GRB_TRY (GrB_Matrix_wait (mat, GrB_MATERIALIZE)) ;
    }

    bool get_mask_struct() { return mask_struct; }

private:

    bool mask_struct{false};
    bool flipxy{false};
    bool mask_comp{false};

    int64_t Annz;
    int64_t Bnnz;
    int64_t Cnnz;
    int64_t N;
    GrB_BinaryOp binop;
    GrB_Monoid  monoid;
    GrB_Semiring  mysemiring;
    std::optional<GB_cuda_mxm_factory> mymxmfactory;
    SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
};
