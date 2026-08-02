
function gbtest149 (ghb)
%GBTEST149 test GrB.expand and GhB.expand.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = pi * ones (3,3) ;

C = gtb_expand (ghb, pi , A) ;

assert (isequal (C, A)) ;
assert (isequal (gtb_type (ghb, C), 'double'))

C = gtb_expand (ghb, pi , A, 'single') ;
assert (isequal (C, single (A))) ;
assert (isequal (gtb_type (ghb, C), 'single'))

fprintf ('gbtest149 (%d): all tests passed\n', ghb) ;

