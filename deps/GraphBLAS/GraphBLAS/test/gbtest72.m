function gbtest72
%GBTEST72 test any-pair semiring

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
dt = struct ('in0', 'transpose') ;
ntrials = 1000 ;

for n = [1 5 10 100 1000]
    nfound = 0 ;
    for trial = 1:ntrials
        x = GrB.random (n, 1, 0.1, 'range', uint32 ([1 255])) ;
        y = GrB.random (n, 1, 0.1, 'range', uint32 ([1 255])) ;
        c1 = x'*y ;

        c3 = GrB.mxm ('+.*', x, y, dt) ;
        assert (isequal (c1, c3)) ;

        c2 = GrB.mxm ('any.pair', x, y, dt) ;

        c1_present = (GrB.entries (c1) == 1) ;
        c2_present = (c2 == 1) ;
        if (c1_present)
            nfound = nfound + 1 ;
        end
        assert (c1_present == c2_present) ;
        assert (c1_present == c2) ;

        c4 = GrB.mxm ('any.oneb', x, y, dt) ;
        assert (isequal (c2, c4)) ;

    end
    fprintf ('n: %4d trials: %4d found: %4d\n', n, ntrials, nfound) ;
end

fprintf ('gbtest72: all tests passed\n') ;

