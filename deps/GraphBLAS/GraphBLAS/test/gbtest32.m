function gbtest32
%GBTEST32 test nonzeros

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

assert (~GrB.burble) ;
rng ('default') ;

for d = 0:.1:1
    for n = 0:10
        A = sprandn (n, n, d) ;
        X = nonzeros (A) ;
assert (~GrB.burble) ;
        G = GrB (A) ;
assert (~GrB.burble) ;
        Y = nonzeros (G) ;
assert (~GrB.burble) ;
        assert (isequal (X, Y)) ;
assert (~GrB.burble) ;
    end
end

fprintf ('gbtest32: all tests passed\n') ;
assert (~GrB.burble) ;

