% test GrB_build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

nthreads_set(2)
grbinfo
test56
test23

test42 ;

nthreads_max = feature_numcores ;
for nthreads = [1 2 4 8 16 20 32 40 64 128 256]
    if (nthreads > 2*nthreads_max)
        break ;
    end
    fprintf ('\n================================================================================\n') ;
    fprintf ('===================== nthreads: %d\n', nthreads) ;
    fprintf ('================================================================================\n') ;
    nthreads_set(nthreads) ;
    test45 ;
end

