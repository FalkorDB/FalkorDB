// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "GpuTimer.h"
#include "GB_cuda_buckets.h"
#include <gtest/gtest.h>
#include "test_data.hpp"
#include "../rmm_wrap/rmm_wrap.hpp"
#include "problem_spec.hpp"

extern "C" {
    #include "GB.h"
}

#include "GB_cuda_common_jitFactory.hpp"
#include "GB_cuda_mxm_dot3_jitFactory.hpp"
#include "GB_cuda_reduce_jitFactory.hpp"
#include "GB_cuda_reduce_factory.hpp"
#include "dataFactory.hpp"

////Operations for test results on CPU
//template<typename T> T myOP_plus( T a, T b) { return  a + b;}
//template<typename T> T myOP_min ( T a, T b) { return  a < b ? a : b;}
//template<typename T> T myOP_max ( T a, T b) { return  a > b ? a : b;}
//template<typename T> T myOP_first ( T a, T b) { return  a ;}
//template<typename T> T myOP_second ( T a, T b) { return  b ;}
//template<typename T> T myOP_times ( T a, T b) { return  a * b ;}
//
//template<typename T> T (*myOpPTR)(T a, T b);
//template<typename T> T (*ADD_ptr)(T a, T b);
//template<typename T> T (*MUL_ptr)(T a, T b);

//AxB_dot3_phase1 kernels
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory( int64_t , int64_t , int64_t , int64_t ) ;

//AxB_dot3_phase2 kernels
template <typename T_C>
bool test_AxB_dot3_phase2_factory( int , int64_t , int64_t , int64_t, int64_t ) ;

template<typename T>
void make_grb_matrix(GrB_Matrix mat, int64_t n_rows, int64_t n_cols,
                     std::vector<int64_t> &indptr,
                     std::vector<int64_t> &indices, T *data,
                     int gxb_sparsity_control = GxB_SPARSE,
                     int gxb_format = GxB_BY_ROW) ;

//Fixture to generate valid inputs and hold them for tests
class AxB_dot3_Test : public ::testing::Test
{
   void SetUp() {}

   void TearDown() {}
};

template<typename T, typename I>
void print_array(void *arr, I size, const char *name) {
    std::cout << "Printing " << name << std::endl;
    for(I i = 0; i < size; ++i) {
        std::cout << static_cast<T*>(arr)[i] << ", ";
    }
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
// test_AxB_phase1_factory: test phase1
//------------------------------------------------------------------------------

// Test generator code, to allow parameterized tests
// Uses jitFactory, dataFactory and GB_jit 
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_phase1_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{

    cudaStream_t stream = (cudaStream_t)rmm_wrap_get_main_stream();

    /********************
     * Launch kernel
     */
    GB_cuda_mxm_factory mysemiringfactory = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mysemiringfactory);

    GpuTimer kernTimer;

    int nthrd = p1lF.get_threads_per_block();
    int ntasks = p1lF.get_number_of_blocks(problem_spec.getM());

    // TODO: Verify that RMM is checking and throwing exceptions
    int nanobuckets_size = NBUCKETS * nthrd * ntasks;
    int blockbuckets_size = NBUCKETS * ntasks;

    int64_t *Nanobuckets = (int64_t*)rmm_wrap_malloc(nanobuckets_size * sizeof (int64_t));
    int64_t *Blockbucket = (int64_t*)rmm_wrap_malloc(blockbuckets_size * sizeof (int64_t));

    kernTimer.Start();
    p1lF.jitGridBlockLaunch(Nanobuckets, Blockbucket,
                            problem_spec.getC(), problem_spec.getM(),
                            problem_spec.getA(), problem_spec.getB(), stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    kernTimer.Stop();
    std::cout<<"returned from phase1 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//  print_array<int64_t>(Nanobuckets, nanobuckets_size, "Nanobuckets");
//  print_array<int64_t>(Blockbucket, blockbuckets_size, "Blockbucket");
    std::cout<<"==== phase1 done=============================" <<std::endl;

    int64_t bucket_count = 0;
    for (int i =0; i< NBUCKETS*ntasks; ++i) bucket_count += Blockbucket[i];
    EXPECT_EQ( bucket_count, problem_spec.getCnnz()); //check we sum to the right structural counts
//
    rmm_wrap_free(Nanobuckets);
    rmm_wrap_free(Blockbucket);

    std::cout << "end phase1 test ------------" << std::endl;

    fflush(stdout);
    return true;
}

// Test generator code, to allow parameterized tests
// Uses jitFactory, dataFactory and GB_jit
template <typename T_C, typename T_M, typename T_A,typename T_B>
bool test_AxB_dense_phase1_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{
    cudaStream_t stream = (cudaStream_t)rmm_wrap_get_main_stream();

    /********************
     * Launch kernel
     */
    GB_cuda_mxm_factory mysemiringfactory = problem_spec.get_mxm_factory();
    dense_phase1launchFactory p1lF(mysemiringfactory);
    p1lF.jitGridBlockLaunch(problem_spec.getC(), problem_spec.getM(), problem_spec.getA(), problem_spec.getB(), stream);
    return true;
}


//------------------------------------------------------------------------------
// test_AxB_phase2_factory: test phase2 and phase2end
//------------------------------------------------------------------------------

template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_AxB_phase2_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec)
{
    cudaStream_t stream = (cudaStream_t)rmm_wrap_get_main_stream();

    auto mymxm = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mymxm);
    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

   GpuTimer kernTimer;
   kernTimer.Start();

   const int64_t mnz = GB_nnz (problem_spec.getM()) ;

   int nthrd = p2lF.get_threads_per_block();
   int ntasks = p2elF.get_number_of_blocks(problem_spec.getM());

    // fabricate data as if it came from phase1:
    int64_t *nanobuckets = (int64_t*)rmm_wrap_malloc(NBUCKETS * nthrd * ntasks * sizeof (int64_t));
    int64_t *blockbucket = (int64_t*)rmm_wrap_malloc(NBUCKETS * ntasks * sizeof (int64_t));
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    int64_t *offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));
    int64_t *bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));

    fillvector_constant(NBUCKETS, bucketp, (int64_t)0);
    fillvector_constant(NBUCKETS, offset, (int64_t)0);
    //fillvector_constant(problem_spec.getCnnz(), bucket, (int64_t)0);

    std::cout << "Running phase1 kernel" << std::endl;
    kernTimer.Start();
    p1lF.jitGridBlockLaunch(nanobuckets, blockbucket,
                            problem_spec.getC(), problem_spec.getM(),
                            problem_spec.getA(), problem_spec.getB(), stream);


    CHECK_CUDA(cudaStreamSynchronize(stream));
    kernTimer.Stop();

    std::cout << " phase1 internal phase2 "<< kernTimer.Elapsed() <<"ms Done." << std::endl;

    //    // launch phase2 (just with p2ntasks as the # of tasks)
    kernTimer.Start();
    p2lF.jitGridBlockLaunch(blockbucket, offset, problem_spec.getM(), stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    kernTimer.Stop();
    std::cout << " phase2 kern "<< kernTimer.Elapsed() <<"ms Done." << std::endl;

//
//    // do the reduction between phase2 and phase2end
    int64_t s= 0;
    for ( int bucket = 0 ; bucket < NBUCKETS+1; ++bucket)
    {
        bucketp[bucket] = s;
        s+= offset[bucket];
    }

    // launch phase2end: note same # of tasks as phase1
    kernTimer.Start();
    p2elF.jitGridBlockLaunch( nanobuckets, blockbucket,
                              bucketp, bucket, offset, problem_spec.getC(),
                              problem_spec.getM(),stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    kernTimer.Stop();
    std::cout<<"returned from phase2end kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;
//
//
    print_array<int64_t>(bucketp, NBUCKETS, "bucketp");
//  print_array<int64_t>(bucket, mnz, "bucket");
    std::cout<<"phase2 done =================="<<std::endl;

    EXPECT_EQ( bucketp[NBUCKETS], problem_spec.getCnnz()); //check we sum to the right structural counts

    rmm_wrap_free(nanobuckets);
    rmm_wrap_free(blockbucket);
    rmm_wrap_free(bucketp);
    rmm_wrap_free(bucket);
    rmm_wrap_free(offset);

    return true;
}

template<typename T>
void make_grb_matrix(GrB_Matrix mat, int64_t n_rows, int64_t n_cols,
                     std::vector<int64_t> &indptr,
                     std::vector<int64_t> &indices, T *data,
                     int gxb_sparsity_control,
                     int gxb_format ) 
{

    GrB_Type type = cuda::jit::to_grb_type<T>();

    GRB_TRY (GrB_Matrix_new (&mat, type, n_rows, n_cols)) ;

    for(int64_t row = 0; row < n_rows; ++row) {
        int64_t start = indptr[row];
        int64_t stop = indptr[row+1];

        for(int64_t offset = start; offset < stop; ++offset) {
            GrB_Index i = (GrB_Index) row;
            GrB_Index j = (GrB_Index) indices[offset];
            T x = data[offset];

            cuda::jit::set_element<T> (mat, x, i, j) ;
        }
    }

    GRB_TRY (GrB_Matrix_wait (mat, GrB_MATERIALIZE)) ;
    GRB_TRY (GB_convert_any_to_non_iso (mat, true)) ;
    GRB_TRY (GxB_Matrix_Option_set (mat, GxB_SPARSITY_CONTROL, gxb_sparsity_control)) ;
    GRB_TRY (GxB_Matrix_Option_set(mat, GxB_FORMAT, gxb_format));


}

template <
    typename T_C, typename T_M, typename T_A,typename T_B,
    typename T_X, typename T_Y, typename T_Z>
bool test_AxB_dot3_sparse_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    // FIXME: Allow the adaptive tests in this guy
    std::cout << "sparse test ======================" << std::endl;

    GpuTimer kernTimer;

    cudaStream_t strm;
    CHECK_CUDA(cudaStreamCreate(&strm));

    std::cout << "sr_code: " << problem_spec.get_mxm_factory().sr_code << std::endl;

    bool result = false;

    int64_t N = problem_spec.getN();
    /**
     * Run Phase 1, phase 2 and phase2end: Compute nanobuckets and blockbuckets
     */

    auto mymxm = problem_spec.get_mxm_factory();
    phase1launchFactory p1lF(mymxm);
    phase2launchFactory p2lF;
    phase2endlaunchFactory p2elF;

    GrB_Matrix C = problem_spec.getC();
    GrB_Matrix M = problem_spec.getM();
    GrB_Matrix A = problem_spec.getA();
    GrB_Matrix B = problem_spec.getB();

    const int64_t mnz = GB_nnz (M) ;
    const int64_t cnz = GB_nnz (C) ;
    const int64_t cvlen = C->vlen ;
    const int64_t cvdim = C->vdim ;
    const int64_t cnvec = C->nvec ;

    bool C_iso = false ;
    int C_sparsity = GB_sparsity (M) ;
    int M_sparsity = GB_sparsity (M) ;
    GrB_Type ctype = problem_spec.getBinaryOp()->ztype ;

    int nthrd = p2lF.get_threads_per_block();
    int ntasks = p2elF.get_number_of_blocks(M);

    // fabricate data as if it came from phase1:
    int64_t *nanobuckets = (int64_t*)rmm_wrap_malloc(NBUCKETS * nthrd * ntasks * sizeof (int64_t));
    int64_t *blockbucket = (int64_t*)rmm_wrap_malloc(NBUCKETS * ntasks * sizeof (int64_t));
    int64_t *bucketp = (int64_t*)rmm_wrap_malloc((NBUCKETS+1) * sizeof (int64_t));
    int64_t *bucket = (int64_t*)rmm_wrap_malloc(mnz * sizeof (int64_t));
    int64_t *offset = (int64_t*)rmm_wrap_malloc(NBUCKETS * sizeof (int64_t));

    fillvector_constant(NBUCKETS, bucketp, (int64_t)0);
    fillvector_constant(NBUCKETS, offset, (int64_t)0);
    //fillvector_constant(problem_spec.getCnnz(), bucket, (int64_t)0);

    std::cout << "sparse phase1 kernel" << std::endl;
    kernTimer.Start();
    p1lF.jitGridBlockLaunch(nanobuckets, blockbucket,
                            C, M, A, B, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"sparse test phase1 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;

    //    // launch phase2 (just with p2ntasks as the # of tasks)
    kernTimer.Start();
    p2lF.jitGridBlockLaunch(blockbucket, offset, M, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"sparse test phase2 kernel "<<kernTimer.Elapsed()<<"ms"<<std::endl;

//
//    // do the reduction between phase2 and phase2end
    int64_t s= 0;
    for ( int bucket = 0 ; bucket < NBUCKETS+1; ++bucket)
    {
        bucketp[bucket] = s;
        s+= offset[bucket];
    }

    std::cout << "Launching phase2end" << std::endl;

    // launch phase2end: note same # of tasks as phase1
    kernTimer.Start();
    p2elF.jitGridBlockLaunch( nanobuckets, blockbucket,
                              bucketp, bucket, offset, C, M, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout << "sparse test phase2end " <<kernTimer.Elapsed()<<"ms"<<std::endl;

    /**
     * Run Phase 3: Execute dot3 on all buckets
     */
    for (int b = 1; b < NBUCKETS; ++b) {// loop on buckets
           int64_t b_start = bucketp[b];
           int64_t b_end = bucketp[b+1];
           int64_t nvecs = b_end - b_start;

           if (nvecs == 0) continue;

           kernTimer.Start();
           phase3launchFactory p3lf(mymxm, (GB_bucket_code)b);
           p3lf.jitGridBlockLaunch( b_start, b_end, bucketp, bucket, C, M,
                                    A, B, strm);
           CHECK_CUDA(cudaStreamSynchronize(strm));

           kernTimer.Stop();
           std::cout << "phase3 bucket="<<b<<" done " <<kernTimer.Elapsed()<<"ms"<<std::endl;
           fflush(stdout);

       }
       C->nzombies += (bucketp[1]); //add pre-zombies to the count;

           GRB_TRY(GrB_Matrix_wait(C, GrB_MATERIALIZE));
           fflush(stdout);

            GrB_Matrix C_expected;
            GrB_Type type = cuda::jit::to_grb_type<T_C>();
            GRB_TRY (GrB_Matrix_new (&C_expected, type, N, N)) ;

            // ensure the GPU is not used
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, -1)) ;
            GB_Global_hack_set (2, 2) ; // hack(2) = 2: never use the GPU

            // Use GrB_DESC_S for structural because dot3 mask will never be complemented
            // The order of B and A is swapped to account for CSR vs CSC assumption
            GRB_TRY (GrB_mxm(C_expected, problem_spec.getM(), NULL, problem_spec.get_semiring(), problem_spec.getB(),
                             problem_spec.getA(), problem_spec.get_mask_struct() ? GrB_DESC_ST1 : GrB_DESC_T1));


            GRB_TRY(GrB_Matrix_wait(C_expected, GrB_MATERIALIZE));

            // compare
            double tol = 0 ;
            GrB_Index nvals1 = 0, nvals2 = 0 ;
            GRB_TRY (GrB_Matrix_nvals (&nvals1, C)) ;
            GRB_TRY (GrB_Matrix_nvals (&nvals2, C_expected)) ;
            if (nvals1 != nvals2) { printf ("Wrong number of nonzeroes found, test fail!!!\n") ; ADD_FAILURE( ) ; }
            GrB_Index nrows, ncols ;
            GrB_Matrix_nrows (&nrows, C_expected) ;
            GrB_Matrix_ncols (&ncols, C_expected) ;

            GrB_Matrix T;

            GRB_TRY (GrB_Matrix_new (&T, GrB_BOOL, nrows, ncols)) ;
            GrB_BinaryOp op = NULL;
            GrB_UnaryOp op_abs = NULL ;
            if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
            else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
            else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
            else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
            else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
            else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
            else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
            else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
            else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
            else if (type == GrB_FP32  )
            {   tol = 1e-6;
                op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
                op_abs = GrB_ABS_FP32 ;
            }
            else if (type == GrB_FP64  )
            {   tol = 1e12;
                op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
                op_abs = GrB_ABS_FP64 ;
            }
            else if (type == GxB_FC32  )
            {   tol = 2e-6;
                op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
                op_abs = GxB_ABS_FC32 ;
            }
            else if (type == GxB_FC64  )
            {   tol = 2e-12;
                op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
                op_abs = GxB_ABS_FC64 ;
            }



            if (tol == 0)
            {
                // check for perfect equality
                GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, C, C_expected,
                    NULL)) ;
                GrB_Index nvals3 = 1 ;
                GRB_TRY (GrB_Matrix_nvals (&nvals3, T)) ;
//                if (nvals1 != nvals3) { printf (" difference matrix wrong size, test fail!!\n") ; ADD_FAILURE( ) ; }
                bool is_same = false ;
                GRB_TRY (GrB_Matrix_reduce_BOOL (&is_same, NULL, GrB_LAND_MONOID_BOOL,
                    T, NULL)) ;
                if (!is_same) { printf (" results don't match, test fail!!\n") ; ADD_FAILURE ( ) ; } 
                GRB_TRY (GrB_Matrix_free (&T)) ;
            }
            else
            {
                // TODO: check with roundoff
                // Diff = C - C_expected
                GrB_Matrix Diff ;
                GRB_TRY (GrB_Matrix_new (&Diff, GrB_FP64, nrows, ncols)) ;
                GRB_TRY (GrB_Matrix_apply (Diff, NULL, NULL, GrB_AINV_FP64, C_expected, NULL)) ;
                GRB_TRY (GrB_Matrix_eWiseAdd_BinaryOp (Diff, NULL, NULL, GrB_PLUS_FP64,
                    C, Diff, NULL)) ;
                GRB_TRY( GrB_Matrix_apply( Diff, NULL, NULL, op_abs, Diff, NULL) );
                GrB_Index nvals3 = 1 ;
                GRB_TRY (GrB_Matrix_nvals (&nvals3, Diff)) ;
                if (nvals1 != nvals3) { printf ("fp difference matrix wrong size, test fail!!\n") ; ADD_FAILURE( ) ; } 
                double is_same = false ;
                GRB_TRY (GrB_Matrix_reduce_FP64 (&is_same, NULL, GrB_PLUS_MONOID_FP64,
                    Diff, NULL)) ;
                printf("difference = %12.6g, rel_l1_err=%12.6g\n", is_same, is_same/nvals3 );
                EXPECT_LT( is_same/nvals3, tol);
                GRB_TRY (GrB_Matrix_free (&Diff)) ;

            }

            // re-enable the GPU
            GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, 0)) ;
            GB_Global_hack_set (2, 1) ; // hack(2) = 1: always use the GPU

    rmm_wrap_free(nanobuckets);
    rmm_wrap_free(blockbucket);
    rmm_wrap_free(bucketp);
    rmm_wrap_free(bucket);
    rmm_wrap_free(offset);
    GRB_TRY(GrB_Matrix_free(&C_expected));
    CHECK_CUDA(cudaStreamDestroy(strm));

    std::cout << "phase 3 test complete ======================" << std::endl;
    return result;
}

template <
        typename T_C, typename T_M, typename T_A,typename T_B,
        typename T_X, typename T_Y, typename T_Z>
bool test_AxB_dot3_dense_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    std::cout << "phase dense  test ======================" << std::endl;

    GpuTimer kernTimer;

    cudaStream_t strm = (cudaStream_t)rmm_wrap_get_main_stream();

    bool result = false;

    int64_t N = problem_spec.getN();

    auto mymxm = problem_spec.get_mxm_factory();
    dense_phase1launchFactory p1lF(mymxm);

    GrB_Matrix C = problem_spec.getC();
    GrB_Matrix M = problem_spec.getM();
    GrB_Matrix A = problem_spec.getA();
    GrB_Matrix B = problem_spec.getB();

    problem_spec.set_sparsity_control( A, GxB_FULL, GxB_BY_ROW);
    problem_spec.set_sparsity_control( B, GxB_FULL, GxB_BY_ROW);

    const int64_t mnz = GB_nnz (M) ;
    const int64_t cnz = GB_nnz (C) ;
    const int64_t cvlen = C->vlen ;
    const int64_t cvdim = C->vdim ;
    const int64_t cnvec = C->nvec ;

    bool C_iso = false ;
    GrB_Type ctype = problem_spec.getBinaryOp()->ztype ;

    std::cout << "Running phase1 kernel" << std::endl;
    kernTimer.Start();
    p1lF.jitGridBlockLaunch(C, M, A, B, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"Dense internal phase1 kernel done "<<kernTimer.Elapsed()<<"ms"<<std::endl;

    std::cout << "Running dense kernel" << std::endl;
    mxm_dense_launchFactory p3lf(mymxm);
    kernTimer.Start();
    p3lf.jitGridBlockLaunch( C, M, A, B, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"Dense kernel done "<<kernTimer.Elapsed()<<"ms"<<std::endl;

    GRB_TRY(GrB_Matrix_wait(C, GrB_MATERIALIZE));
    fflush(stdout);

    GrB_Matrix C_expected;
    GrB_Type type = cuda::jit::to_grb_type<T_C>();
    GRB_TRY (GrB_Matrix_new (&C_expected, type, N, N)) ;

    // ensure the GPU is not used
    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, -1)) ;
    GB_Global_hack_set (2, 2) ; // hack(2) = 2: never use the GPU

    // Use GrB_DESC_S for structural because dot3 mask will never be complemented
    // The order of B and A is swapped to account for CSR vs CSC assumption
    GRB_TRY (GrB_mxm(C_expected, problem_spec.getM(), NULL, problem_spec.get_semiring(), problem_spec.getB(),
                     problem_spec.getA(), problem_spec.get_mask_struct() ? GrB_DESC_ST1 : GrB_DESC_T1));

    GRB_TRY(GrB_Matrix_wait(C_expected, GrB_MATERIALIZE));
    std::cout << "nnz: " << GB_nnz (C_expected) << std::endl ;

    // compare
    double tol = 0 ;
    GrB_Index nvals1 = 0, nvals2 = 0 ;
    GRB_TRY (GrB_Matrix_nvals (&nvals1, C)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals2, C_expected)) ;
    if (nvals1 != nvals2) { printf ("Wrong number of nonzeroes found, test fail!!! nvals1=%lu, nvals2=%lu\n", nvals1, nvals2) ; ADD_FAILURE( ) ; }
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, C_expected) ;
    GrB_Matrix_ncols (&ncols, C_expected) ;

    GrB_Matrix T;

    GRB_TRY (GrB_Matrix_new (&T, GrB_BOOL, nrows, ncols)) ;
    GrB_BinaryOp op = NULL;
    GrB_UnaryOp op_abs = NULL ;
    if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
    else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
    else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
    else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
    else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
    else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
    else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
    else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
    else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
    else if (type == GrB_FP32  )
    {   tol = 5e-6;
        op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
        op_abs = GrB_ABS_FP32 ;
    }
    else if (type == GrB_FP64  )
    {   tol = 1e12;
        op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
        op_abs = GrB_ABS_FP64 ;
    }
    else if (type == GxB_FC32  )
    {   tol = 2e-6;
        op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
        op_abs = GxB_ABS_FC32 ;
    }
    else if (type == GxB_FC64  )
    {   tol = 2e-12;
        op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
        op_abs = GxB_ABS_FC64 ;
    }



    if (tol == 0)
    {
        // check for perfect equality
        GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, C, C_expected,
                                                NULL)) ;
        GrB_Index nvals3 = 1 ;
        GRB_TRY (GrB_Matrix_nvals (&nvals3, T)) ;
//        if (nvals1 != nvals3) { printf (" difference matrix wrong size, test fail!! nvals1=%ld nvals3=%ld\n", nvals1, nvals3) ; ADD_FAILURE( ) ; }
        bool is_same = false ;
        GRB_TRY (GrB_Matrix_reduce_BOOL (&is_same, NULL, GrB_LAND_MONOID_BOOL,
                                         T, NULL)) ;
        if (!is_same) { printf (" results don't match, test fail!!\n") ; ADD_FAILURE ( ) ; }
        GRB_TRY (GrB_Matrix_free (&T)) ;
    }
    else
    {
        // TODO: check with roundoff
        // Diff = C - C_expected
        GrB_Matrix Diff ;
        GRB_TRY (GrB_Matrix_new (&Diff, GrB_FP64, nrows, ncols)) ;
        GRB_TRY (GrB_Matrix_apply (Diff, NULL, NULL, GrB_AINV_FP64, C_expected, NULL)) ;
        GRB_TRY (GrB_Matrix_eWiseAdd_BinaryOp (Diff, NULL, NULL, GrB_PLUS_FP64,
                                               C, Diff, NULL)) ;
        GRB_TRY( GrB_Matrix_apply( Diff, NULL, NULL, op_abs, Diff, NULL) );
        GrB_Index nvals3 = 1 ;
        GRB_TRY (GrB_Matrix_nvals (&nvals3, Diff)) ;
        if (nvals1 != nvals3) { printf ("fp difference matrix wrong size, test fail!!\n") ; ADD_FAILURE( ) ; }
        double is_same = false ;
        GRB_TRY (GrB_Matrix_reduce_FP64 (&is_same, NULL, GrB_PLUS_MONOID_FP64,
                                         Diff, NULL)) ;
        printf("difference = %12.6g, rel_l1_err=%12.6g\n", is_same, is_same/nvals3 );
        EXPECT_LT( is_same/nvals3, tol);
        GRB_TRY (GrB_Matrix_free (&Diff)) ;

    }

    // re-enable the GPU
    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, 0)) ;
    GB_Global_hack_set (2, 1) ; // hack(2) = 1: always use the GPU


    GRB_TRY(GrB_Matrix_free(&C_expected));

    std::cout << "phase 3 dense test complete ======================" << std::endl;
    return result;
}

template <
        typename T_C, typename T_M, typename T_A,typename T_B,
        typename T_X, typename T_Y, typename T_Z>
bool test_AxB_dot3_sparse_dense_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    std::cout << "sparse dense test ======================" << std::endl;

    GpuTimer kernTimer;

    cudaStream_t strm;
    CHECK_CUDA(cudaStreamCreate(&strm));

    bool result = false;

    int64_t N = problem_spec.getN();

    GrB_Matrix C = problem_spec.getC();
    GrB_Matrix M = problem_spec.getM();
    GrB_Matrix A = problem_spec.getA();
    GrB_Matrix B = problem_spec.getB();

    problem_spec.set_sparsity_control( A, GxB_SPARSE, GxB_BY_ROW);

    // TODO: Need to make sure the format itself is actually dense.
    problem_spec.set_sparsity_control( B, GxB_FULL, GxB_BY_ROW);

    auto mymxm = problem_spec.get_mxm_factory();
    dense_phase1launchFactory p1lF(mymxm);

    const int64_t mnz = GB_nnz (M) ;
    const int64_t cnz = GB_nnz (C) ;
    const int64_t cvlen = C->vlen ;
    const int64_t cvdim = C->vdim ;
    const int64_t cnvec = C->nvec ;

    bool C_iso = false ;
    GrB_Type ctype = problem_spec.getBinaryOp()->ztype ;

    std::cout << "Running dense_phase1 kernel" << std::endl;
    kernTimer.Start();
    p1lF.jitGridBlockLaunch(C, M, A, B, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"Dense internal phase1 kernel done "<<kernTimer.Elapsed()<<"ms"<<std::endl;

    std::cout << "Running sparse dense kernel" << std::endl;
    mxm_sparse_dense_launchFactory spdnlf(mymxm);
    kernTimer.Start();
    spdnlf.jitGridBlockLaunch( C, M, A, B, strm);
    CHECK_CUDA(cudaStreamSynchronize(strm));
    kernTimer.Stop();
    std::cout<<"Sparse_Dense done "<<kernTimer.Elapsed()<<"ms"<<std::endl;

    GRB_TRY(GrB_Matrix_wait(C, GrB_MATERIALIZE));
    fflush(stdout);

    GrB_Matrix C_expected;
    GrB_Type type = cuda::jit::to_grb_type<T_C>();
    GRB_TRY (GrB_Matrix_new (&C_expected, type, N, N)) ;

    // ensure the GPU is not used
    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, -1)) ;
    GB_Global_hack_set (2, 2) ; // hack(2) = 2: never use the GPU

    // Use GrB_DESC_S for structural because dot3 mask will never be complemented
    // The order of B and A is swapped to account for CSR vs CSC assumption
    GRB_TRY (GrB_mxm(C_expected, problem_spec.getM(), NULL, problem_spec.get_semiring(), problem_spec.getB(),
                     problem_spec.getA(), problem_spec.get_mask_struct() ? GrB_DESC_ST1 : GrB_DESC_T1));

    GRB_TRY(GrB_Matrix_wait(C_expected, GrB_MATERIALIZE));
    std::cout << "nnz: " << GB_nnz (C_expected) << std::endl ;

    // compare
    double tol = 0 ;
    GrB_Index nvals1 = 0, nvals2 = 0 ;
    GRB_TRY (GrB_Matrix_nvals (&nvals1, C)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals2, C_expected)) ;
    if (nvals1 != nvals2) { printf ("Wrong number of nonzeroes found, test fail!!! nvals1=%lu, nvals2=%lu\n", nvals1, nvals2) ; ADD_FAILURE( ) ; }
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, C_expected) ;
    GrB_Matrix_ncols (&ncols, C_expected) ;

    GrB_Matrix T;

    GRB_TRY (GrB_Matrix_new (&T, GrB_BOOL, nrows, ncols)) ;
    GrB_BinaryOp op = NULL;
    GrB_UnaryOp op_abs = NULL ;
    if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
    else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
    else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
    else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
    else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
    else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
    else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
    else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
    else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
    else if (type == GrB_FP32  )
    {   tol = 5e-6;
        op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
        op_abs = GrB_ABS_FP32 ;
    }
    else if (type == GrB_FP64  )
    {   tol = 1e12;
        op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
        op_abs = GrB_ABS_FP64 ;
    }
    else if (type == GxB_FC32  )
    {   tol = 2e-6;
        op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
        op_abs = GxB_ABS_FC32 ;
    }
    else if (type == GxB_FC64  )
    {   tol = 2e-12;
        op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
        op_abs = GxB_ABS_FC64 ;
    }



    if (tol == 0)
    {
        // check for perfect equality
        GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, op, C, C_expected,
                                                NULL)) ;
        GrB_Index nvals3 = 1 ;
        GRB_TRY (GrB_Matrix_nvals (&nvals3, T)) ;
//        if (nvals1 != nvals3) { printf (" difference matrix wrong size, test fail!! nvals1=%ld nvals3=%ld\n", nvals1, nvals3) ; ADD_FAILURE( ) ; }
        bool is_same = false ;
        GRB_TRY (GrB_Matrix_reduce_BOOL (&is_same, NULL, GrB_LAND_MONOID_BOOL,
                                         T, NULL)) ;
        if (!is_same) { printf (" results don't match, test fail!!\n") ; ADD_FAILURE ( ) ; }
        GRB_TRY (GrB_Matrix_free (&T)) ;
    }
    else
    {
        // TODO: check with roundoff
        // Diff = C - C_expected
        GrB_Matrix Diff ;
        GRB_TRY (GrB_Matrix_new (&Diff, GrB_FP64, nrows, ncols)) ;
        GRB_TRY (GrB_Matrix_apply (Diff, NULL, NULL, GrB_AINV_FP64, C_expected, NULL)) ;
        GRB_TRY (GrB_Matrix_eWiseAdd_BinaryOp (Diff, NULL, NULL, GrB_PLUS_FP64,
                                               C, Diff, NULL)) ;
        GRB_TRY( GrB_Matrix_apply( Diff, NULL, NULL, op_abs, Diff, NULL) );
        GrB_Index nvals3 = 1 ;
        GRB_TRY (GrB_Matrix_nvals (&nvals3, Diff)) ;
        if (nvals1 != nvals3) { printf ("fp difference matrix wrong size, test fail!!\n") ; ADD_FAILURE( ) ; }
        double is_same = false ;
        GRB_TRY (GrB_Matrix_reduce_FP64 (&is_same, NULL, GrB_PLUS_MONOID_FP64,
                                         Diff, NULL)) ;
        printf("difference = %12.6g, rel_l1_err=%12.6g\n", is_same, is_same/nvals3 );
        EXPECT_LT( is_same/nvals3, tol);
        GRB_TRY (GrB_Matrix_free (&Diff)) ;

    }

    // re-enable the GPU
    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, 0)) ;
    GB_Global_hack_set (2, 1) ; // hack(2) = 1: always use the GPU


    GRB_TRY(GrB_Matrix_free(&C_expected));
    CHECK_CUDA(cudaStreamDestroy(strm));

    std::cout << "phase 3 dense test complete ======================" << std::endl;
    return result;
}


template <typename T_C, typename T_M, typename T_A, typename T_B>
bool test_reduce_factory(mxm_problem_spec<T_C, T_M, T_A, T_B> &problem_spec) {

    std::cout << "reduce test ======================" << std::endl;

    // TODO: This test doesn't really fit the `mxm` category
    GrB_Monoid monoid = problem_spec.getMonoid();
    int64_t N = problem_spec.getN();

    GrB_Matrix A;

    // TODO: Using C here so that the reduced type matches
    GrB_Matrix_dup(&A, problem_spec.getC());
    GrB_Type type = cuda::jit::to_grb_type<T_C>();

    A->i[0] = GB_FLIP(A->i[0]); // FIXME
    A->i[1] = GB_FLIP(A->i[1]); // FIXME
    A->nzombies = 2; // FIXME: use an opaque method to insert zombies into A

    //GRB_TRY (GxB_Matrix_fprint (A, "A", GxB_SHORT_VERBOSE, stdout)) ;

    GB_cuda_reduce_factory myreducefactory;
    myreducefactory.reduce_factory(monoid, A);

    T_C actual;
    GB_cuda_reduce(myreducefactory, A, &actual, monoid );

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, -1)) ;
    GB_Global_hack_set (2, 2) ; // hack(2) = 2: never use the GPU

    T_C expected;
    GRB_TRY(cuda::jit::matrix_reduce(&expected, A, monoid));

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, 0)) ;
    GB_Global_hack_set (2, 1) ; // hack(2) = 1: always use the GPU

    double tol = 0;
    GrB_BinaryOp op = NULL;
    GrB_UnaryOp op_abs = NULL ;

    if      (type == GrB_BOOL  ) op = GrB_EQ_BOOL   ;
    else if (type == GrB_INT8  ) op = GrB_EQ_INT8   ;
    else if (type == GrB_INT16 ) op = GrB_EQ_INT16  ;
    else if (type == GrB_INT32 ) op = GrB_EQ_INT32  ;
    else if (type == GrB_INT64 ) op = GrB_EQ_INT64  ;
    else if (type == GrB_UINT8 ) op = GrB_EQ_UINT8  ;
    else if (type == GrB_UINT16) op = GrB_EQ_UINT16 ;
    else if (type == GrB_UINT32) op = GrB_EQ_UINT32 ;
    else if (type == GrB_UINT64) op = GrB_EQ_UINT64 ;
    else if (type == GrB_FP32  )
    {   tol = 1e-6;
        op = (tol == 0)? GrB_EQ_FP32 : GrB_MINUS_FP32   ;
        op_abs = GrB_ABS_FP32 ;
    }
    else if (type == GrB_FP64  )
    {   tol = 1e12;
        op = (tol == 0)? GrB_EQ_FP64 : GrB_MINUS_FP64   ;
        op_abs = GrB_ABS_FP64 ;
    }
    else if (type == GxB_FC32  )
    {   tol = 2e-6;
        op = (tol == 0)? GxB_EQ_FC32 : GxB_MINUS_FC32   ;
        op_abs = GxB_ABS_FC32 ;
    }
    else if (type == GxB_FC64  )
    {   tol = 2e-12;
        op = (tol == 0)? GxB_EQ_FC64 : GxB_MINUS_FC64   ;
        op_abs = GxB_ABS_FC64 ;
    }

    if(tol == 0) {
       EXPECT_EQ( actual , expected);
        //std::cout << "results do not match: reduced=" << expected << ", actual=" << actual << std::endl;
        //exit(1);
    } else if ( (tol > 0) && ( ( type ==GrB_FP32) || ( type ==GxB_FC32) 
                            || ( type ==GrB_FP64) || ( type ==GxB_FC64) ) ){
       EXPECT_LT( abs((double)actual - (double)expected)/(abs((double)expected)+1.e-12), tol) ;
    }

    std::cout<< expected<< " " << actual<< "reduce test complete ======================" << std::endl;
    GRB_TRY(GrB_Matrix_free(&A));

    return expected == actual;
}

