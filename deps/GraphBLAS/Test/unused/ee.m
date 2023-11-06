%EE eWiseMult and eWiseAdd performance tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
addpath ('~/ssget') ;
addpath ('spok') ;

for threads = 1:4
    nthreads_set (threads) ;
    grbinfo
    test68  % eWiseMult performance
    test58(0)  % eWiseAdd performance
    test61  % eWiseMult performance
    test39  % eWiseAdd performance
end

