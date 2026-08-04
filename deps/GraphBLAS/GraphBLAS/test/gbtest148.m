
function gbtest148 (ghb)
%GBTEST148 test GrB/log and GhB/log.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = rand (3) ;
G = gtb (ghb, A, 'complex') ;

C1 = log (A) ;
C2 = log (G) ;
err = norm (C1 - C2, 1) ;
assert (err < 1e-10) ;
assert (isequal (gtb_type (ghb, C2), 'double'))

C1 = log10 (A) ;
C2 = log10 (G) ;
err = norm (C1 - C2, 1) ;
assert (err < 1e-10) ;
assert (isequal (gtb_type (ghb, C2), 'double'))

C1 = log2 (A) ;
C2 = log2 (G) ;
err = norm (C1 - C2, 1) ;
assert (err < 1e-10) ;
assert (isequal (gtb_type (ghb, C2), 'double'))

C1 = sqrt (A) ;
C2 = sqrt (G) ;
err = norm (C1 - C2, 1) ;
assert (err < 1e-10) ;
assert (isequal (gtb_type (ghb, C2), 'double'))

fprintf ('gbtest148 (%d): all tests passed\n', ghb) ;

