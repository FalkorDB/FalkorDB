function gbtest8
%GBTEST8 test GrB.select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%   tril
%   triu
%   diag
%   offdiag

%   nonzero     ~=0
%   zero        ==0
%   positive    >0
%   nonnegative >=0
%   ltzero      <0
%   lezero      <=0

%   ~=
%   ==
%   >
%   >=
%   <
%   <=

rng ('default') ;
n = 5 ;
m = 8 ;
A = sparse (10 * rand (m,n) - 5) .* sprand (m, n, 0.8) ;
M = logical (sprand (m, n, 0.5)) ;

b = 0.5 ;
desc.kind = 'sparse' ;

A (1,1) = b ;
A (2,2) = -b ;
A (3,4) = b ;

%-------------------------------------------------------------------------
% tril
%-------------------------------------------------------------------------

    C1 = tril (A) ;
    C2 = GrB.select ('tril', A, 0) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = tril (A, k) ;
        C2 = GrB.select ('tril', A, k) ;
        C3 = GrB.select ('tril', A, k, desc) ;
        assert (gbtest_eq (C1, C2))
        assert (gbtest_eq (C1, C3))
        assert (isequal (class (C3), 'double')) ;
    end

    C1 = tril (A, 0) ;
    C2 = GrB.select ('tril', A, 0) ;
    assert (gbtest_eq (C1, C2))

    C1 = A + tril (A, 0) ;
    C2 = GrB.select (A, '+', 'tril', A, 0) ;
    assert (gbtest_eq (C1, C2))

    C1 = A ;
    T = A + tril (A, 0) ;
    C1 (M) = T (M) ;
    C2 = GrB.select (A, M, '+', 'tril', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% triu
%-------------------------------------------------------------------------

    C1 = triu (A) ;
    C2 = GrB.select ('triu', A, 0) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = triu (A, k) ;
        C2 = GrB.select ('triu', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = triu (A, 0) ;
    C2 = GrB.select ('triu', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% diag
%-------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = GrB.select ('diag', A, 0) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = A .* spdiags (ones (d,1), k, m, n) ;
        C2 = GrB.select ('diag', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = GrB.select ('diag', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% offdiag
%-------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = GrB.select ('offdiag', A, 0) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = A .* (1 - spdiags (ones (d,1), k, m, n)) ;
        C2 = GrB.select ('offdiag', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = GrB.select ('offdiag', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% nonzero
%-------------------------------------------------------------------------

    % all explicit entries in the built-in sparse matrix are nonzero,
    % so this does nothing.  A better test would be to compute a GraphBLAS
    % matrix with explicit zeros first.

    M = (A ~= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('nonzero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('~=0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '~=0') ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% zero
%-------------------------------------------------------------------------

    % all explicit entries in the built-in sparse matrix are nonzero,
    % so this does nothing.

    C1 = sparse (m,n) ;

    C2 = GrB.select ('zero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('==0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '==0') ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% positive
%-------------------------------------------------------------------------

    M = (A > 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('positive', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('>0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '>0') ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('positive.double', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% nonnegative
%-------------------------------------------------------------------------

    M = (A >= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('nonnegative', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('>=0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '>=0') ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% negative
%-------------------------------------------------------------------------

    M = (A < 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('negative', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('<0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '<0') ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% nonpositive
%-------------------------------------------------------------------------

    M = (A <= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('nonpositive', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select ('<=0', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '<=0') ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% ~=
%-------------------------------------------------------------------------

    M = (A ~= b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('~=', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '~=', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% ==
%-------------------------------------------------------------------------

    M = (A == b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('==', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '==', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% >
%-------------------------------------------------------------------------

    M = (A > b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('>', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '>', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% >=
%-------------------------------------------------------------------------

    M = (A >= b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('>=', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '>=', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% <
%-------------------------------------------------------------------------

    M = (A < b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('<', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '<', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% <=
%-------------------------------------------------------------------------

    M = (A <= b) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = GrB.select ('<=', A, b) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (A, '<=', b) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------
% gtzero, with mask and accum
%-------------------------------------------------------------------------

    Cin = sprand (m, n, 0.5) ;
    C2 = GrB.select (Cin, '+', '>0', A) ;
    C1 = Cin ;
    C1 (A > 0) = C1 (A > 0) + A (A > 0) ; 
    assert (gbtest_eq (C1, C2))

    M = logical (sprand (m, n, 0.5)) ;
    Cin = sprand (m, n, 0.5) ;
    C2 = GrB.select (Cin, M, '>0', A) ;
    C1 = Cin ;
    T = sparse (m, n) ;
    T (A > 0) = A (A > 0) ;
    C1 (M) = T (M) ;
    assert (gbtest_eq (C1, C2))

    C2 = GrB.select (Cin, M, '+', '>0', A) ;
    C1 = Cin ;
    T = sparse (m, n) ;
    T (A > 0) = A (A > 0) ;
    C1 (M) = C1 (M) + T (M) ;
    assert (gbtest_eq (C1, C2))

fprintf ('gbtest8: all tests passed\n') ;

