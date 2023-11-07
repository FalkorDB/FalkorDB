function test181
%TEST181 test transpose with explicit zeros in the Mask

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% see also test99

rng ('default') ;

fprintf ('test181: transpose with explicit zeros in the Mask\n') ;

masks = { '', 'complement', 'structural', 'structural complement' } ;
repls = { '', 'replace' } ;

for n = 10:20
    for d = 0:.1:1
        fprintf ('.') ;
        for k = 1:10

            C = sprand (n, n, 0.1) ;
            A = GB_spec_random (n, n, 0.1) ;
            A.sparsity = 4 ;
            if (d == 1)
                Mask = sparse (rand (n)) ;
            else
                Mask = sprand (n, n, d) ;
            end
            if (nnz (Mask) > 0)
                [i j x] = find (Mask) ;
                nz = length (x) ;
                p = randperm (nz, floor(nz/2)) ;
                x (p) = 0 ;
                i = uint64 (i-1) ;
                j = uint64 (j-1) ;
                Mask = GB_mex_Matrix_build (i,j,x,n,n,[]) ;
                Mask = Mask.matrix ;
            end

            for k1 = 1:length(masks)
                for k2 = 1:2

                    clear desc
                    desc.inp0 = 'tran' ;
                    if (~isempty (masks {k1}))
                        desc.mask = masks {k1} ;
                    end
                    if (~isempty (repls {k2}))
                        desc.outp = repls {k2} ;
                    end

                    C2 = GB_spec_transpose (C, Mask, [], A, desc) ;
                    C3 = GB_mex_transpose  (C, Mask, [], A, desc) ;
                    GB_spec_compare (C2, C3) ;

                    C2 = GB_spec_transpose (C, Mask, [], C, desc) ;
                    C3 = GB_mex_transpose  (C, Mask, [], C, desc) ;
                    GB_spec_compare (C2, C3) ;

                end
            end
        end
    end
end

fprintf ('\ntest181: all tests passed\n') ;

