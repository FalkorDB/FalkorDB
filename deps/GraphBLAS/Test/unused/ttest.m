% test GrB_extractTuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
addpath ('~/ssget') ;
addpath ('spok') ;

for nth = [2 1]
    nthreads_set (nth) ;

    debug_on
    grbinfo

    test11
    test16
    test35
    test40
    testc9

    debug_off
    grbinfo

    test11
    test16
    test35
    test40
    testc9

end
