function gbtest95
%GBTEST95 test indexing

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = gb_octave ;
G = GrB.empty (GrB ([0 2])) ;
assert (isequal (size (G), [0 2])) ;

A = magic (4) ;
I = GrB ([1 2]) ;
G = GrB (A) ;
X = G (:,1) ;
Y = G (:,1) ;
C1 = X (I) ;
C2 = Y ([1 2]) ;
assert (isequal (C1, C2)) ;

C1 = X ({ I }) ;
assert (isequal (C1, C2)) ;

C1 = G ({ }, { })  ;
assert (isequal (C1, G)) ;

H = GrB (2^59, 2^60) ;
[m, n] = size (H) ;
s = GrB.isfull (H) ;
assert (~s) ;
assert (isequal ([m n], [2^59 2^60])) ;
assert (isa ([m n], 'int64')) ;

H = GrB.random (3, 4, inf, 'range', GrB ([2 4], 'int8')) ;
assert (GrB.isfull (H)) ;
assert (isequal (GrB.type (H), 'int8')) ;

H = GrB.random (H, 'range', GrB ([3 4], 'uint32')) ;
assert (GrB.isfull (H)) ;
assert (isequal (GrB.type (H), 'uint32')) ;

C = tril (H, GrB (1,1)) ;
assert (istril (C)) ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    if (gb_contains (type, 'complex') || isequal (type, 'logical'))
        continue ;
    end
    I = GrB ([1 2], type) ;
    if (have_octave)
        % octave: indices into built-in matrices cannot be objects
        I = int64 (I) ;
    end
    C1 = A (I,I) ;
    C2 = A ([1 2], [1 2]) ;
    C3 = A (int8 ([1 2]), int8 ([1 2])) ;
    C4 = G (I,I) ;
    assert (isequal (C1, C2))
    assert (isequal (C1, C3))
    assert (isequal (C1, C4))
end

if (~have_octave)
    % octave: indices into built-in matrices cannot be objects
    I1 = [1 2 ; 3 4] ;
    I2 = GrB (I1) ;
    C1 = A (I1,I1) ;
    C2 = A (I2,I2) ;
    H = GrB (2^60, 2^60) ;
    H (1:2,1:2) = I1 ;
    C3 = A (H,H) ;
    assert (isequal (C1, C2))
    assert (isequal (C1, C3))
end

A = [-1 2] ;
B = [2 0.5] ;
C1 = A.^B ;
C2 = GrB (A).^B ;
assert (isequal (C1, C2))
assert (isreal (C2)) 

fprintf ('gbtest95: all tests passed\n') ;

