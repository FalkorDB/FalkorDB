function gbtest120
%GBTEST120 test subsref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

x = sparse (1:5) ;
C1 = x (:) ;
y = GrB (x) ;
C2 = y (:) ;
assert (isequal (C1, C2)) ;

x = sparse (magic (4)) ;
C1 = x (:) ;
y = GrB (x) ;
C2 = y (:) ;
assert (isequal (C1, C2)) ;

% linear indexing would require a 128-bit integer, so it fails
n = 2^50 ;
H = GrB (n,n) ;
H (1,1) = 42 ;
H (n,n) = 99 ;
H
try
    C = H (:)
    ok = false ;
catch expected_error
    % 'problem too large'
    ok = true ;
end
assert (ok)
expected_error

fprintf ('gbtest120: all tests passed\n') ;

