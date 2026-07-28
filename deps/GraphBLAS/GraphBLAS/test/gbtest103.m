function gbtest103 (ghb)
%GBTEST103 test iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 2^52 ;
A = gtb_ones (ghb, n,n)  %#ok<NOPRT>
assert (A (n/2, n) == 1) ;

nz = gtb_nvals (ghb, A) ;
assert (nz == n^2) ;

fprintf ('\ngbtest103 (%d): all tests passed\n', ghb) ;

