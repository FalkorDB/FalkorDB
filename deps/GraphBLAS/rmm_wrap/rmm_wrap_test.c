//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap_test.c:  simple main program for testing rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "rmm_wrap.h"

int main()
{

    size_t init_size, max_size, stream_pool_size = 1;
    init_size = 256*(1ULL<<10);
    max_size  = 256*(1ULL<<20);

    //printf(" pool init size %ld, max size %ld\n", init_size, max_size);
    rmm_wrap_initialize_all_same( rmm_wrap_managed, init_size, max_size, stream_pool_size );
    printf("RMM initialized!  in managed mode\n");

    void *p;
    size_t buff_size = (1ULL<<13)+152;

    printf(" asked for %ld", buff_size);
    fflush(stdout);
    p = (void *)rmm_wrap_allocate( &buff_size );
    printf(" actually allocated  %ld\n", buff_size);
    fflush(stdout);
    rmm_wrap_deallocate( p, buff_size);
    rmm_wrap_finalize();

    rmm_wrap_initialize_all_same(rmm_wrap_device, init_size, max_size, stream_pool_size );
    printf("RMM initialized!  in device mode\n");

    buff_size = (1ULL<<13)+157;
    printf(" asked for %ld", buff_size);
    fflush(stdout);
    p = (void *)rmm_wrap_allocate( &buff_size );
    printf(" actually allocated  %ld\n", buff_size);
    fflush(stdout);
    rmm_wrap_deallocate( p, buff_size);
    rmm_wrap_finalize();
}

