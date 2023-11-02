function test141
%TEST141 test GrB_eWiseAdd (all types and operators) for dense matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
types = types.all ;

fprintf ('test141 ------------ GrB_eWiseAdd with dense matrices\n') ;

m = 5 ;
n = 5 ;

rng ('default') ;

M = sprand (m, n, 0.5) ;

Amat2 = 2 * sparse (rand (m,n)) ;
Bmat2 = 2 * sparse (rand (m,n)) ;
Cmat2 = 2 * sparse (rand (m,n)) ;

Amat = 50 * sparse (rand (m,n)) ;
Bmat = 50 * sparse (rand (m,n)) ;
Cmat = 50 * sparse (rand (m,n)) ;

Emat = sprand (m, n, 0.5) ;
Smat = sparse (m,n) ;
desc.mask = 'structural' ;

A.matrix = Amat ; A.class = 'see below' ;
B.matrix = Bmat ; B.class = 'see below' ;
C.matrix = Cmat ; C.class = 'see below' ;
S.matrix = Smat ; S.class = 'see below' ;
E.matrix = Emat ; E.class = 'see below' ;

A.sparsity = 8 ;
B.sparsity = 8 ;
C.sparsity = 8 ;

for k2 = 1:length(binops)
    binop = binops {k2}  ;
    if (isequal (binop, 'pow'))
        continue ;
    end
    fprintf ('\n%-10s ', binop) ;

    for k1 = 1:length (types)
        type = types {k1}  ;
        op.opname = binop ;
        op.optype = type ;

        try
            GB_spec_operator (op) ;
        catch
            continue ;
        end

        fprintf ('.') ;

        switch (binop)
            % domain is ok, but limit it to avoid integer typecast
            % failures from O(eps) errors, or overflow to inf
            case { 'ldexp', 'pow' }
                A.matrix = Amat2 ;
                B.matrix = Bmat2 ;
                C.matrix = Cmat2 ;
            otherwise
                % no change
        end

        A.class = type ;
        B.class = type ;
        E.class = type ;

        if (k2 > 20)
            % eq, ne, gt, lt, ge, le
            S.class = 'logical' ;
            C.class = 'logical' ;
        else
            S.class = type ;
            C.class = type ;
        end

        id = test_cast (0, type) ;

        %---------------------------------------
        % C = A+B
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (S, [ ], [ ], op, A, B, [ ]) ;
        C1 = GB_mex_Matrix_eWiseAdd  (S, [ ], [ ], op, A, B, [ ]) ;
        GB_spec_compare (C0, C1, id) ;

        %---------------------------------------
        % C<M> = A+B, both A and B dense
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (S, M, [ ], op, A, B, desc) ;
        C1 = GB_mex_Matrix_eWiseAdd  (S, M, [ ], op, A, B, desc) ;
        GB_spec_compare (C0, C1, id) ;

        %---------------------------------------
        % C<M> = A+E, A dense, E sparse
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (S, M, [ ], op, A, E, desc) ;
        C1 = GB_mex_Matrix_eWiseAdd  (S, M, [ ], op, A, E, desc) ;
        GB_spec_compare (C0, C1, id) ;

        %---------------------------------------
        % C<M> = E+A, A dense, E sparse
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (S, M, [ ], op, E, A, desc) ;
        C1 = GB_mex_Matrix_eWiseAdd  (S, M, [ ], op, E, A, desc) ;
        GB_spec_compare (C0, C1, id) ;

        %---------------------------------------
        % C += A+B
        %---------------------------------------

        if (~GB_spec_is_positional (op))
            C0 = GB_spec_Matrix_eWiseAdd (C, [ ], op, op, A, B, [ ]) ;
            C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], op, op, A, B, [ ]) ;
            GB_spec_compare (C0, C1, id) ;
        end
    end
end

fprintf ('\ntest141: all tests passed\n') ;

