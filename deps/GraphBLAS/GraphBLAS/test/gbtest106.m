function gbtest106 (ghb)
%GBTEST106 test build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = gtb_build (ghb, 1:5, 1:5, true, 5, 5, 'xor') ;
assert (isequal (A, logical (speye (5)))) ;

fprintf ('\ngbtest106 (%d): all tests passed\n', ghb) ;

