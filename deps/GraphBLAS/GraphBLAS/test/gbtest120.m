function gbtest120 (ghb)
%GBTEST120 test subsref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

x = sparse (1:5) ;
C1 = x (:) ;
y = gtb (ghb, x) ;
C2 = y (:) ;
assert (isequal (C1, C2)) ;

x = sparse (magic (4)) ;
C1 = x (:) ;
y = gtb (ghb, x) ;
C2 = y (:) ;
assert (isequal (C1, C2)) ;

% linear indexing would require a 128-bit integer, so it fails
n = 2^50 ;
H = gtb (ghb, n,n) ;
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

fprintf ('gbtest120 (%d): all tests passed\n', ghb) ;

