function gbtest58 (ghb)
%GBTEST58 test uplus

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = 1 - 2 * rand (3) ;
G = gtb (ghb, A) ;
G = +G ;
A = +A ;

assert (isequal (A, G)) ;

fprintf ('gbtest58 (%d): all tests passed\n', ghb) ;

