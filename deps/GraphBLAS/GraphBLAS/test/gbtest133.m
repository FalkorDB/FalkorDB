function gbtest133
%GBTEST133 test GhB.apply (simple inplace usage only)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

A = magic (4) ;
C = GhB (A) ;

GhB.apply (C, 'sqrt', C) ;
assert (isequal (C, sqrt (A))) ;
assert (isequal (class (C), 'GhB')) ; 

GhB.apply (C, '-', C) ;
assert (isequal (C, -sqrt (A))) ;
assert (isequal (class (C), 'GhB')) ; 

clear C
assert (GhB.nmalloc == 0) ;

fprintf ('gbtest133: all tests passed\n') ;

