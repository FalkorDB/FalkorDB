%TESTALL3 run testall with different # of threads

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
grbinfo
for k = [1 2 4]
    nthreads_set (k) ;
    debug_off ;
    testall 
    debug_on
    testall 
end
