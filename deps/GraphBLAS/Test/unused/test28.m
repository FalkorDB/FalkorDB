function test28
%TEST28 test mxm with aliased inputs, C<C> = accum(C,C*C)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

seed = 1 ;
for n = [1 5 10 100]

    for trial = 1:30

        C = GB_mex_random (n, n, 10*n, 0, seed) ; seed = seed + 1 ;

        C1 = GB_mex_mxm_alias (C, 'plus', semiring, [ ]) ;
        C2 = GB_mex_mxm (C, C, 'plus', semiring, C, C, [ ]) ;
        assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;
    end
end

fprintf ('test28: mxm alias tests passed\n') ;

