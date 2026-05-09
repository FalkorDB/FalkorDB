//------------------------------------------------------------------------------
// rmm_wrap/rmm_wrap_test.c:  simple main program for testing rmm_wrap
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "rmm_wrap.h"

int main()
{

    uint64_t init_memsize, max_memsize ;
    init_memsize = 256*(1ULL<<10);
    max_memsize  = 256*(1ULL<<20);

    rmm_wrap_initialize_all_same( rmm_wrap_managed, init_memsize, max_memsize);
    printf("RMM initialized!  in managed mode\n");

    void *p;
    size_t buff_memsize = (1ULL<<13)+152;

    printf(" asked for %ld", buff_memsize);
    fflush(stdout);
    p = (void *)rmm_wrap_allocate( &buff_memsize );
    printf(" actually allocated  %ld\n", buff_memsize);
    fflush(stdout);
    rmm_wrap_deallocate( p, buff_memsize);
    rmm_wrap_finalize();

    rmm_wrap_initialize_all_same(rmm_wrap_device, init_memsize, max_memsize);
    printf("RMM initialized!  in device mode\n");

    buff_memsize = (1ULL<<13)+157;
    printf(" asked for %ld", buff_memsize);
    fflush(stdout);
    p = (void *)rmm_wrap_allocate( &buff_memsize );
    printf(" actually allocated  %ld\n", buff_memsize);
    fflush(stdout);
    rmm_wrap_deallocate( p, buff_memsize);
    rmm_wrap_finalize();
}

