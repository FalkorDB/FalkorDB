function test69
%TEST69 test GrB_assign with aliased inputs, C<C>(:,:) = accum(C(:,:),C)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test69 ------------------  assign alias tests\n') ;

rng ('default') ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

desc = struct ('outp', 'replace') ;

seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        fprintf ('.') ;

        for trial = 1:30
            A = GB_mex_random (m, n, 10*n, 0, seed) ; seed = seed + 1 ;
            C = GB_mex_random (m, n, 10*n, 0, seed) ; seed = seed + 1 ;

            % C<C> += C
            C1 = GB_mex_assign_alias (C, 'plus', [ ], [ ], [ ]) ;
            C2 = GB_mex_assign (C, [ ], 'plus', C, [ ], [ ], [ ], 0) ;
            assert (isequal (C1, C2)) ;

            % C<C,replace> += C
            C1 = GB_mex_assign_alias (C, 'plus', [ ], [ ], desc) ;
            C2 = GB_mex_assign (C, [ ], 'plus', C, [ ], [ ], desc, 0) ;
            assert (isequal (C1, C2)) ;

            % C<C,replace> = C
            C1 = GB_mex_assign_alias (C, [ ], [ ], [ ], desc) ;
            C2 = GB_mex_assign (C, [ ], [ ], C, [ ], [ ], desc, 0) ;
            C2.iso = 0 ;
            assert (isequal (C1, C2)) ;

            % C(I,J)<C> += C(I,J)
            I = uint64 (randperm (m) - 1) ;
            J = uint64 (randperm (n) - 1) ;
            C1 = GB_mex_assign_alias (C, 'plus', I, J, [ ]) ;
            C2 = GB_mex_assign (C, [ ], 'plus', C, I, J, [ ], 0) ;
            assert (isequal (C1, C2)) ;

            % C<C,replace> += C
            C1 = GB_mex_subassign_alias (C, 'plus', desc) ;
            C2 = GB_mex_subassign (C, C, 'plus', C, [ ], [ ], desc) ;
            assert (isequal (C1, C2)) ;

            % C<C,replace> = C
            C1 = GB_mex_subassign_alias (C, [ ], desc) ;
            C2 = GB_mex_subassign (C, C, [ ], C, [ ], [ ], desc) ;
            assert (isequal (C1, C2)) ;

            % C(:,:) = 0
            Z = GB_mex_expand (sparse (1), 0) ;
            C1 = GB_mex_subassign (C, [ ], [ ], Z, [ ], [ ], desc) ;
            C2 = sparse (m, n) ;
            assert (isequal (1 * C1.matrix, C2)) ;

        end
    end
end

fprintf ('\ntest69: assign alias tests passed\n') ;

