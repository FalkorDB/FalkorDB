function test04
%TEST04 test and demo for accumulator/mask and transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n-------------------- simple mask and transpose tests\n') ;

rng ('default') ;
C = rand (4)
Z = magic (4)
Mask = mod (Z,2) == 0

for C_replace = [false true]
    for Mask_comp = [false true]
        fprintf ('Use Mask.  C_replace: %d Mask_comp: %d\n', ...
            C_replace, Mask_comp) ;
        Cresult = GB_spec_mask (C, Mask, Z, C_replace, Mask_comp) ;
        % C2 =  apply_mask_sparse (C, Z, Mask, C_replace, Mask_comp) ;
        % assert (isequal (Cresult, C2))

        D = [ ] ;
        if (Mask_comp)
            D.mask = 'complement' ;
        end
        if (C_replace)
            D.outp = 'replace' ;
        end

        A = Z ;
        fprintf ('C3 <Mask> = C + A'' :\n') ;
        C3 = GB_spec_transpose (C, Mask, 'plus', A, D) ;
        C5 = GB_mex_transpose  (sparse(C), sparse(Mask), 'plus', sparse(A), D);
        assert (isequal (C3.matrix, C5.matrix))

    end
end

for C_replace = [false true]
    for Mask_comp = [false true]
        fprintf ('No Mask.  C_replace: %d Mask_comp: %d\n', ...
            C_replace, Mask_comp) ;
        Cresult = GB_spec_mask (C, [ ], Z, C_replace, Mask_comp) ;
        % C2 = apply_mask_sparse  (C, Z, [ ], C_replace, Mask_comp) ;
        % assert (isequal (Cresult, C2))

        D = [ ] ;
        if (Mask_comp)
            D.mask = 'complement' ;
        end
        if (C_replace)
            D.outp = 'replace' ;
        end

        A = Z ;
        fprintf ('C3 <no mask complement:%d replace:%d> = C + A'' :\n', ...
            Mask_comp, C_replace) ;
D
        C3 = GB_spec_transpose (C, [ ], 'plus', A, D) ;
        C5 = GB_mex_transpose  (sparse(C), [ ], 'plus', sparse(A), D);
        assert (isequal (C3.matrix, C5.matrix))
    end
end

fprintf ('\ntest04: all tests passed\n') ;

