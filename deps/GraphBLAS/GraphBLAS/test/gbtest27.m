function gbtest27
%GBTEST27 test conversion to full
% This test does a lot of typecasting and requires either many JIT kernels (248
% of them if all default FactoryKernels are enabled) or uses many generic
% methods if the JIT is disabled.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = gbtest_types ;

rng ('default') ;
for k1 = 1:length (types)

    atype = types {k1} ;
    A = 100 * sprand (3, 3, 0.5) ;
    H = full (A, 'double', GrB (0)) ;
    assert (norm (H-A,1) == 0)
    B = A ;
    B (A == 0) = 1 ; %#ok<*SPRIX>
    H = full (A, 'double', GrB (1)) ;
    assert (norm (H-B,1) == 0)

    F = rand (3) ;
    H = full (F, 'double', GrB (0)) ;
    assert (norm (H-F,1) == 0)
    assert (isa (H, 'GrB'))

    H = GrB (A, atype) ;
    G = full (H) ;
    assert (GrB.entries (G) == prod (size (G))) ;

    for k2 = 1:length (types)

        fprintf ('.') ;
        gtype = types {k2} ;
        G = full (H, gtype) ;
        K = full (G, atype) ;
        for id = [0 1 inf]
            C = full (H, gtype, id) ; %#ok<*NASGU>
        end

        assert (GrB.entries (G) == prod (size (G))) ; %#ok<*PSIZE>
    end
end

fprintf ('\ngbtest27: all tests passed\n') ;

