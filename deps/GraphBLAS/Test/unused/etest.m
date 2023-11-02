% test eWise

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
addpath ('~/ssget') ;
addpath ('spok') ;

for k = [2 1]
    nthreads_set (k) ;
    grbinfo

    test15
    test72
    testc8
    test68  % eWiseMult performance
    test105
    % test20
    % test26
    test34
    test58  % eWiseAdd performance

    test61  % eWiseMult performance
    test18
    test39  % eWiseAdd performance

    test111 % eWiseAdd performance
end

