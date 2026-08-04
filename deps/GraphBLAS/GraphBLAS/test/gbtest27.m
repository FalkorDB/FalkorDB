function gbtest27 (ghb)
%GBTEST27 test conversion to full
% This test does a lot of typecasting and requires either many JIT kernels (248
% of them if all default FactoryKernels are enabled) or uses many generic
% methods if the JIT is disabled.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;
desc = struct ;

types = gbtest_types ;

for k1 = 1:length (types)

    atype = types {k1} ;
    A = 100 * sprand (3, 3, 0.5) ;
    H = full (A, 'double', gtb (ghb, 0)) ;
    assert (norm (H-A,1) == 0)
    H = full (GrB (A), 'double', GrB (0)) ;
    assert (norm (H-A,1) == 0)
    H = full (GrB (A), 'double', GrB (0)) ;
    assert (norm (H-A,1) == 0)
    H = full (GrB (A), 'double') ;
    assert (norm (H-A,1) == 0)
    H = gzb_full (0, GrB (A), 'double', GrB (0), desc) ;
    assert (norm (H-A,1) == 0)
    H = gzb_full (0, GrB (A), 'double') ;
    assert (norm (H-A,1) == 0)

    B = gtb (ghb, A) ;
    B (A == 0) = 1 ; %#ok<*SPRIX>
    H = full (A, 'double', gtb (ghb, 1)) ;
    assert (norm (H-B,1) == 0)

    F = rand (3) ;
    H = full (F, 'double', gtb (ghb, 0)) ;
    assert (norm (H-F,1) == 0)
    assert (gbtest_isa (ghb, H))

    H = gtb (ghb, A, atype) ;
    G = full (H) ;
    assert (gtb_entries (ghb, G) == prod (size (G))) ;

    for k2 = 1:length (types)

        fprintf ('.') ;
        gtype = types {k2} ;
        G = full (H, gtype) ;
        K = full (G, atype) ;
        for id = [0 1 inf]
            C = full (H, gtype, id) ; %#ok<*NASGU>
        end

        assert (gtb_entries (ghb, G) == prod (size (G))) ; %#ok<*PSIZE>
    end
end

fprintf ('\ngbtest27 (%d): all tests passed\n', ghb) ;

