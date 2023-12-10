function gbtest11
%GBTEST11 test GrB, sparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = 100 * rand (4) ;
A (1,1:2) = 0 %#ok<*NOPRT>
S = sparse (A)

  x1 = GrB (S)
  x2 = full (x1)
  x3 = double (x2)
  assert (gbtest_eq (S, x3))

% assert (gbtest_eq (S, double (full (GrB (S)))))

  x1 = GrB (S)
  x2 = full (x1)
  x3 = full (x2)
  x4 = double (x3)
  assert (gbtest_eq (S, x4))

% assert (gbtest_eq (S, double (full (full (GrB (S))))))

assert (gbtest_eq (S, double (full (double (full (GrB (S)))))))

S2 = double (GrB (full (double (full (GrB (S))))))
assert (norm (S-S2,1) == 0)
% S2 = 1*S2 ;
assert (gbtest_eq (S, S2))

S2 = double (GrB (double (GrB (full (double (full (GrB (S))))))))
assert (gbtest_eq (S, S2))

S = logical (S) ;
assert (gbtest_eq (S, full (GrB (S))))

X = int8 (A)
G = GrB (X)
assert (gbtest_eq (X, full (int8 (G))))
assert (gbtest_eq (X, int8 (full (G))))

X = int16 (A)
G = GrB (X)
assert (gbtest_eq (X, full (int16 (G))))
assert (gbtest_eq (X, int16 (full (G))))

X = int32 (A)
G = GrB (X)
assert (gbtest_eq (X, full (int32 (G))))
assert (gbtest_eq (X, int32 (full (G))))

X = int64 (A)
G = GrB (X)
assert (gbtest_eq (X, full (int64 (G))))
assert (gbtest_eq (X, int64 (full (G))))

X = uint8 (A)
G = GrB (X)
assert (gbtest_eq (X, full (uint8 (G))))
assert (gbtest_eq (X, uint8 (full (G))))

X = uint16 (A)
G = GrB (X)
assert (gbtest_eq (X, full (uint16 (G))))
assert (gbtest_eq (X, uint16 (full (G))))

X = uint32 (A)
G = GrB (X)
assert (gbtest_eq (X, full (uint32 (G))))
assert (gbtest_eq (X, uint32 (full (G))))

X = uint64 (A)
G = GrB (X)
full (G)
assert (gbtest_eq (X, full (uint64 (G))))
assert (gbtest_eq (X, uint64 (full (G))))

B = 100 * rand (4) ;
B (1,[1 3]) = 0 ;

have_octave = gb_octave ;
X = complex (A)
G = GrB (X)
if (have_octave)
    % the octave7, full(...) function converts its result to real if the
    % imaginary part is zero, but MATLAB and GraphBLAS return as complex.
    assert (gbtest_eq (X, G)) ;
else
    assert (gbtest_eq (X, full (complex (G))))
    assert (gbtest_eq (X, complex (full (G))))
end

X = complex (A,B)
G = GrB (X)
assert (gbtest_eq (X, full (complex (G))))
assert (gbtest_eq (X, complex (full (G))))

X = rand (4) ;
Y = GrB (X) ;
Z = sparse (Y) ;
W = sparse (Z) ;
assert (gbtest_eq (X, Z)) ;
assert (gbtest_eq (X, Y)) ;
assert (gbtest_eq (X, W)) ;

S = struct (Y) ;
Z = GrB (S) ;
assert (gbtest_eq (Z, Y)) ;

assert (GrB.isfull (Z)) ;
assert (GrB.isfull (double (Z))) ;
assert (~GrB.isfull (speye (3))) ;
assert (~GrB.isfull (GrB (speye (3)))) ;

fprintf ('gbtest11: all tests passed\n') ;

