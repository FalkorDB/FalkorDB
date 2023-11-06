function test173
%TEST173 test GrB_assign C<A>=A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = { 'logical', 'double', 'double complex' } ;

m = 10 ;
n = 14 ;

rng ('default') ;

desc.mask = 'structural' ;

for k = 1:length (types)

    ctype = types {k} ;
    fprintf ('%s, ', ctype) ;

    for d = [0.5 inf]

        C = GB_spec_random (m, n, d, 100, ctype) ;
        C = GB_spec_matrix (C) ;
        C.matrix = sparse (C.matrix) ;

        A = GB_spec_random (m, n, 0.5, 100, ctype) ;
        A.matrix = sparse (A.matrix) ;
        A_nonzero = full (A.matrix ~= 0) ;

        A_dense = GB_spec_random (m, n, inf, 100, ctype) ;
        A_dense = GB_spec_matrix (A_dense) ;
        A_dense.matrix = sparse (A_dense.matrix) ;
        A_dense_nonzero = full (A_dense.matrix ~= 0) ;

        A_iso = A ;
        A_iso.matrix = spones (A.matrix) ;
        A_iso_nonzero = full (A_iso.matrix ~= 0) ;
        A_iso.iso = true ;

        for C_sparsity = 1:15
            C.sparsity = C_sparsity ;

            for A_sparsity = 1:15
                A.sparsity = A_sparsity ;
                A_iso.sparsity = A_sparsity ;
                A_dense.sparsity = A_sparsity ;

                % C<A> = A
                C1 = GB_mex_assign_alias_mask (C, A, [ ]) ;
                C2 = full (C.matrix) ;
                C2 (A_nonzero) = full (A.matrix (A_nonzero)) ;
                err = norm (double (C2) - double (C1.matrix), 1) ;
                assert (err == 0) ;

                % C<A,struct> = A
                B = A ;
                B.matrix = sparse (B.matrix) ;
                C3 = GB_mex_assign_alias_mask (C, B, desc) ;
                err = norm (double (C2) - double (C3.matrix), 1) ;
                assert (err == 0) ;

                % C<A> = A where A is iso
                C1 = GB_mex_assign_alias_mask (C, A_iso, [ ]) ;
                C2 = full (C.matrix) ;
                C2 (A_iso_nonzero) = full (A_iso.matrix (A_iso_nonzero)) ;
                err = norm (double (C2) - double (C1.matrix), 1) ;
                assert (err == 0) ;

                % C<A,struct> = A where A is dense
                C1 = GB_mex_assign_alias_mask (C, A_dense, desc) ;
                err = norm (double (A_dense.matrix) - double (C1.matrix), 1) ;
                assert (err == 0) ;

                % C<A> = A where A is dense
                C1 = GB_mex_assign_alias_mask (C, A_dense, [ ]) ;
                err = norm (double (A_dense.matrix) - double (C1.matrix), 1) ;
                assert (err == 0) ;

            end
        end
    end
end

fprintf ('\ntest173: all tests passed\n') ;

