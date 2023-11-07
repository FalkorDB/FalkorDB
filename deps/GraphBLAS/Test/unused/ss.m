% test GxB_select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all ; make

for k = 32 % [1 2 4 8 16 32]
    nthreads_set (k)
    debug_off
    grbinfo
    test27  % LoHi_band, for user-defined
    debug_on
    grbinfo
    test27
end

