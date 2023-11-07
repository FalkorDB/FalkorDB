function test180
%TEST180 subassign and assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% For test coverage, this test must be run with 1 thread

fprintf ('test180: --------------------------------- assign\n') ;

n = 20 ;
rng ('default') ;

Cin = GB_spec_random (n, n, 0.5, 1, 'double') ;
m = 6 ;
I1 = [2 3 5 1 9 11] ;
J1 = [4 5 1 9 2 12] ;
I0 = uint64 (I1) - 1 ;
J0 = uint64 (J1) - 1 ;
M1.matrix = logical (sprand (m, m, 0.5)) ;
M2.matrix = logical (sprand (n, n, 0.5)) ;
A = GB_spec_random (m, m, 0.8, 1, 'double') ;

dr = struct ('outp', 'replace') ;
dc = struct ('mask', 'complement') ;
drc = struct ('outp', 'replace', 'mask', 'complement') ;

for c = 1:15
    Cin.sparsity = c ;
    fprintf ('.') ;
    for a = 1:15
        A.sparsity = a ;

        % C(I,J) = A
        C1 = GB_spec_subassign (Cin, [ ], [ ], A, I1, J1, [ ], false) ;
        C2 = GB_mex_subassign  (Cin, [ ], [ ], A, I0, J0, [ ]) ;
        GB_spec_compare (C1, C2) ;

        for ms = [2 4]
            M1.sparsity = ms ;
            M2.sparsity = ms ;

            % C(I,J)<M> = A
            C1 = GB_spec_subassign (Cin, M1, [ ], A, I1, J1, [ ], false) ;
            C2 = GB_mex_subassign  (Cin, M1, [ ], A, I0, J0, [ ]) ;
            GB_spec_compare (C1, C2) ;

            % C<M,repl>(I,J) = A
            C1 = GB_spec_assign (Cin, M2, [ ], A, I1, J1, dr, false) ;
            C2 = GB_mex_assign  (Cin, M2, [ ], A, I0, J0, dr) ;
            GB_spec_compare (C1, C2) ;

            % C<!M,repl>(I,J) = A
            C1 = GB_spec_assign (Cin, M2, [ ], A, I1, J1, drc, false) ;
            C2 = GB_mex_assign  (Cin, M2, [ ], A, I0, J0, drc) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<M,repl> = A
            C1 = GB_spec_subassign (Cin, M1, [ ], A, I1, J1, dr, false) ;
            C2 = GB_mex_subassign  (Cin, M1, [ ], A, I0, J0, dr) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<M,repl> += A
            C1 = GB_spec_subassign (Cin, M1, 'plus', A, I1, J1, dr, false) ;
            C2 = GB_mex_subassign  (Cin, M1, 'plus', A, I0, J0, dr) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<M,repl> = scalar
            C1 = GB_spec_subassign (Cin, M1, [ ], 3, I1, J1, dr, true) ;
            C2 = GB_mex_subassign  (Cin, M1, [ ], 3, I0, J0, dr) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<!M> = scalar
            C1 = GB_spec_subassign (Cin, M1, [ ], 3, I1, J1, dc, true) ;
            C2 = GB_mex_subassign  (Cin, M1, [ ], 3, I0, J0, dc) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<!M> += scalar
            C1 = GB_spec_subassign (Cin, M1, 'plus', 3, I1, J1, dc, true) ;
            C2 = GB_mex_subassign  (Cin, M1, 'plus', 3, I0, J0, dc) ;
            GB_spec_compare (C1, C2) ;

            % C(I,J)<!M,repl> += scalar
            C1 = GB_spec_subassign (Cin, M1, 'plus', 3, I1, J1, drc, true) ;
            C2 = GB_mex_subassign  (Cin, M1, 'plus', 3, I0, J0, drc) ;
            GB_spec_compare (C1, C2) ;
        end

    end
end

fprintf ('\ntest180: all tests passed\n') ;

