
// References:
// https://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
// https://sortingalgos.miraheze.org/wiki/Bitonic_Sort

#include "sort/GB_sort.h"

//------------------------------------------------------------------------------
// GB_bitonic: bitonic sort
//------------------------------------------------------------------------------

GrB_Info GB_bitonic
(
    int32_t *restrict A,    // array of size n
    int64_t n,              // n does not need to be a power of 2
    int nthreads
)
{
    printf ("GB_bitonic: n %ld, nthreads %d\n", n, nthreads) ;

//  for CUDA variant:
//  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
//  int nthreads = blockDim.x * gridDim.x ;
//  int64_t oops = 0 ;

    int64_t Nhalf = n/2 ;
    for (int64_t k = 2, stage = 1 ; k < 2*n ; k = k << 1, stage++)
    {
        bool dir = (((((n-1) >> stage) + 1) & 1) != 0) ;
        for (int64_t j = k >> 1 ; j > 0 ; j = j >> 1)
        {
            uint64_t mask = j-1 ;

            // for CUDA variant:
            // parallel loop for all threads in the threadblock:
//          for (int64_t ipair = tid ; ipair < Nhalf ; ipair += nthreads)

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int64_t ipair = 0 ; ipair < Nhalf ; ipair++)
            {
                // Consider the pair of entries A [ileft] and A [iright] where
                // ileft < iright always holds.  The ileft entry is obtained by
                // inserting a 0-bit in ipair, where the lower bits of ipair
                // (in the mask) are kept and the upper bits are shifted to the
                // left by one.  For example, when j = 8, mask = 7 = 0111 in
                // binary, then ileft is obtained by shifting the upper bits
                // (all but the lower 3) of ipair to the left by one bit
                // position, inserting a zero bit.  Thus, if ipair = 1101111
                // and j=8 then ileft = 1101o111 where o = 0 denotes the
                // inserted bit in ileft.  Then iright is obtained by setting
                // the o bit to 1.  Thus ileft < iright always holds.
                int64_t ileft = ((ipair & ~mask) << 1) | (ipair & mask) ;
                int64_t iright = ileft | j ;
                // ensure that A [iright] is in the range A [0..n-1]
                if (iright >= n)
                {
                    /* oops++ ; */
                    continue ;
                }
                // if desc is true, swap descending, else swap ascending
                bool desc = (((ileft & k) != 0) == dir) ;
                int aleft  = A [ileft] ;
                int aright = A [iright] ;
                if (desc ? (aleft < aright) : (aleft > aright))
                {
                    // swap A [ileft] and A [iright]
                    A [ileft ] = aright ;
                    A [iright] = aleft ;
                }
            }
            // for CUDA variant:
            // syncthreads here
        }
    }

    // printf ("oops: %ld\n", oops) ;
    return (GrB_SUCCESS) ;
}

