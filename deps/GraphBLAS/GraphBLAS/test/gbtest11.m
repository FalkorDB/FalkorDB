function gbtest11 (ghb)
%GBTEST11 test GrB, GhB, sparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = 100 * rand (4) ;
A (1,1:2) = 0 %#ok<*NOPRT>
S = sparse (A)

  x1 = gtb (ghb, S)
  x2 = full (x1)
  x3 = double (x2)
  assert (gbtest_eq (S, x3))

% assert (gbtest_eq (S, double (full (gtb (ghb, S)))))

  x1 = gtb (ghb, S)
  x2 = full (x1)
  x3 = full (x2)
  x4 = double (x3)
  assert (gbtest_eq (S, x4))

% assert (gbtest_eq (S, double (full (full (gtb (ghb, S))))))

assert (gbtest_eq (S, double (full (double (full (gtb (ghb, S)))))))

S2 = double (gtb (ghb, full (double (full (gtb (ghb, S))))))
assert (norm (S-S2,1) == 0)
% S2 = 1*S2 ;
assert (gbtest_eq (S, S2))

S2 = double (gtb (ghb, double (gtb (ghb, full (double (full (gtb (ghb, S))))))))
assert (gbtest_eq (S, S2))

S = logical (S) ;
assert (gbtest_eq (S, full (gtb (ghb, S))))

X = int8 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (int8 (G))))
assert (gbtest_eq (X, int8 (full (G))))

X = int16 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (int16 (G))))
assert (gbtest_eq (X, int16 (full (G))))

X = int32 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (int32 (G))))
assert (gbtest_eq (X, int32 (full (G))))

X = int64 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (int64 (G))))
assert (gbtest_eq (X, int64 (full (G))))

X = uint8 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (uint8 (G))))
assert (gbtest_eq (X, uint8 (full (G))))

X = uint16 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (uint16 (G))))
assert (gbtest_eq (X, uint16 (full (G))))

X = uint32 (A)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (uint32 (G))))
assert (gbtest_eq (X, uint32 (full (G))))

X = uint64 (A)
G = gtb (ghb, X)
full (G)
assert (gbtest_eq (X, full (uint64 (G))))
assert (gbtest_eq (X, uint64 (full (G))))

B = 100 * rand (4) ;
B (1,[1 3]) = 0 ;

have_octave = gb_octave ;
X = complex (A)
G = gtb (ghb, X)
if (have_octave)
    % the builtin octave F=full(A) function converts F to real if the imaginary
    % part of A is zero, but MATLAB and GraphBLAS return F as complex with zero
    % imaginary part.
    assert (gbtest_eq (X, G)) ;
else
    assert (gbtest_eq (X, full (complex (G))))
    assert (gbtest_eq (X, complex (full (G))))
end

X = complex (A,B)
G = gtb (ghb, X)
assert (gbtest_eq (X, full (complex (G))))
assert (gbtest_eq (X, complex (full (G))))

X = rand (4) ;
Y = gtb (ghb, X) ;
Z = sparse (Y) ;
W = sparse (Z) ;
assert (gbtest_eq (X, Z)) ;
assert (gbtest_eq (X, Y)) ;
assert (gbtest_eq (X, W)) ;

assert (gtb_isfull (ghb, Z)) ;
assert (gb_isfull (GrB (Z))) ;
assert (gtb_isfull (ghb, double (Z))) ;
assert (~gtb_isfull (ghb, speye (3))) ;
assert (~gb_isfull (GrB (speye (3)))) ;
assert (~gtb_isfull (ghb, gtb (ghb, speye (3)))) ;

fprintf ('gbtest11 (%d): all tests passed\n', ghb) ;

