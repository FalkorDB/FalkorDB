% test GxB_select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make

for k = [1 2 4 8 ] % 16 32]
    nthreads_set (k)

    debug_on
    grbinfo
    stat

    test129 %GxB_select (tril and nonzero, hypersparse)
    test134 % shorter version of test25
    test27  % LoHi_band, for user-defined
    test26  % performance test
    test76  % resize
    test25  % built-in, exhaustive test

    debug_off
    grbinfo
    stat

    test129 %GxB_select (tril and nonzero, hypersparse)
    test134 % shorter version of test25
    test27  % LoHi_band, for user-defined
    test26  % performance test
    test76  % resize
    test25  % built-in, exhaustive test

end

