function test286
%TEST286 test kron with idxop

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sprand (2, 3, 0.5) ;
B = sprand (3, 4, 0.5) ;
for atrans = 0:1
    for btrans = 0:1
        C2 = GB_spec_kron_idx (A, B, atrans, btrans) ;
        for csc = 0:1
%           GrB.burble (1) ;
            C = GB_mex_kron_idx (A, B, atrans, btrans, csc) ;
%           GrB.burble (0) ;
            assert (isequal (C, C2)) ;
        end
    end
end

fprintf ('\ntest286: all tests passed\n') ;

