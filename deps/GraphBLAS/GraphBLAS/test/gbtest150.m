function gbtest150 (ghb)
%GBTEST150 test [GrB,GhB].wait

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = sparse (10, 10) ;
C = gtb (ghb, 10, 10)

C (1:4,1:4) = magic (4)
A (1:4,1:4) = magic (4)

gtb_wait (ghb, C) ;
C

assert (isequal (A, C)) ;

fprintf ('gbtest150 (%d): all tests passed\n', ghb) ;

