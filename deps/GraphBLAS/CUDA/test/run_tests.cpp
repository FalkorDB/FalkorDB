#include <gtest/gtest.h>

#include "GraphBLAS.h"
#include "rmm_wrap.h"

#include "test_utility.hpp"

int main(int argc, char **argv) {

    size_t init_size, max_size, stream_pool_size;
    init_size = 256*(1ULL<<10);
    max_size  = 256*(1ULL<<20);
    stream_pool_size = 1;

    printf(" pool init size %ld, max size %ld\n", init_size, max_size);
    rmm_wrap_initialize_all_same( rmm_wrap_managed, init_size, max_size,  stream_pool_size);

    GRB_TRY (GxB_init (GxB_NONBLOCKING_GPU,
        rmm_wrap_malloc, rmm_wrap_calloc, rmm_wrap_realloc, rmm_wrap_free)) ;

    std::cout << "Done initializing graphblas and rmm" << std::endl;

    GRB_TRY (GxB_Global_Option_set (GxB_GLOBAL_GPU_ID, 0)) ;

    size_t buff_size = (1ULL<<13)+152;
    void *p = (void *)rmm_wrap_allocate( &buff_size );

    ::testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();

    rmm_wrap_deallocate( p, buff_size);
    GRB_TRY (GrB_finalize());
    rmm_wrap_finalize();
    std::cout << "Tests complete" << std::endl;

    return r;
}
