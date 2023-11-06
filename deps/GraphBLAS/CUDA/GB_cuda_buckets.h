//------------------------------------------------------------------------------
// GB_cuda_buckets.h: definitions for buckets using for dot3 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd only in the GraphBLAS/CUDA/GB_cuda*.cu source files.

#ifndef GB_CUDA_BUCKETS_H
#define GB_CUDA_BUCKETS_H

#define NBUCKETS 3

// NBUCKETS buckets: computed by up to NBUCKETS-1 kernel launches (zombies need
// no work...), each using different kernels (with different configurations
// depending on the bucket).

// dot3:  C<M>=A'B, M is sparse or hyper, C is sparse or hyper
// 32 kernels A,B: (hyper,sparse,bitmap,full)^2 x (M and C are sparse/hyper)

typedef enum
{
    GB_BUCKET_ZOMBIE = 0,       // C(i,j) is a zombie (not a bucket)
    // both A and B are sparse/hyper:
    GB_BUCKET_VSVS = 1,         // vsvs: both A(:,i) and B(:,j) are very sparse
    GB_BUCKET_MERGEPATH = 2,    // mp: use the merge-path method
    // A is sparse/hyper and B is bitmap/full, or
    // A is bitmap/full  and B is sparse/hyper
    GB_BUCKET_VSDN = 1,         // vsdn: the sparse vector is very sparse
    GB_BUCKET_SPDN = 2,         // spdn: sparse vector has lots of entries;
                                // use a whole warp for each dot product
}
GB_bucket_code ;

// These may use another bucket enum:

    // two full/(sparse,hyper) kernels:
    //  // CUDA kernel: spdn, handles 4 buckets:
    //  // A(:,i) is dense and B(:,j) is very sparse (< 256 entries)
    //  GB_BUCKET_DNVS = 2,
    //  // A(:,i) is dense and B(:,j) is sparse (>= 256 entries)
    //  GB_BUCKET_DNSP = 3,

    // a sparse/full kernel
    //  // A(:,i) is very sparse (< 256 entries) and B(:,j) is dense
    //  GB_BUCKET_VSDN = 4,
    //  // A(:,i) is sparse (>= 256 entries) and B(:,j) is dense
    //  GB_BUCKET_SPDN = 5,

    // a sparse/bitmap kernel
    // a bitmap/bitmap kernel
    // a bitmap/sparse kernel
    // ...

#endif
