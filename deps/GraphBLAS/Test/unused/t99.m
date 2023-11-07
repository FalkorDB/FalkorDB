
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

clear all
test99
t {1} = [4 1] ;
logstat ('test99',t) ;  % GB_mex_transpose with explicit zeros in the Mask

