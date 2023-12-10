%TESTALL2 run testall with different # of threads

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make
for k = [4 1] %  8 20]

    nthreads_set (k,1) ;

    % if (k == 1)
        debug_on
        testall 
    % end

    debug_off ;
    testall 
end

mtest
