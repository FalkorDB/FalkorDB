function test142
%TEST142 test GrB_assign for dense matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
types = types.all ;

fprintf ('test142 ------------ GrB_assign with dense matrices\n') ;

m = 10 ;
n = 12 ;

rng ('default') ;

M = sprand (m, n, 0.5) ;

Amat2 = sparse (2 * rand (m,n)) ;
Bmat2 = sparse (2 * sprand (m,n, 0.5)) ;
Cmat2 = sparse (2 * rand (m,n)) ;

Amat = 50 * Amat2 ;
Bmat = 50 * Bmat2 ;
Cmat = 50 * Cmat2 ;

Smat = sparse (m,n) ;
Xmat = sparse (pi) ;
desc.mask = 'structural' ;
drep.outp = 'replace' ;

A.matrix = Amat ; A.class = 'see below' ;
B.matrix = Bmat ; B.class = 'see below' ;
C.matrix = Cmat ; C.class = 'see below' ;
S.matrix = Smat ; S.class = 'see below' ;
X.matrix = Xmat ; X.class = 'see below' ;
Bmask = logical (Bmat) ;
A.sparsity = 8 ;
C.sparsity = 8 ;
X.sparsity = 8 ;

for k1 = 1:length (types)
    type = types {k1}  ;
    fprintf ('%s ', type) ;

    A.class = type ;
    id = test_cast (0, type) ;
    A_iso = A ;
    A_iso.iso = true ;

    for k3 = 1:3

        if (k3 == 1)
            X.class = type ;
            B.class = type ;
            C.class = 'logical' ;
            S.class = 'logical' ;
        elseif (k3 == 2)
            X.class = type ;
            B.class = type ;
            C.class = type ;
            S.class = type ;
        else
            X.class = 'int8' ;
            B.class = 'int8' ;
            C.class = type ;
            S.class = type ;
        end

        %---------------------------------------
        % C<M> = A where A is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], A, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, M, [ ], A, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], B, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, M, [ ], B, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = A where A is dense and C starts empty
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], A, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign  (S, M, [ ], A, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = A where A is iso full and C starts empty
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], A_iso, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign  (S, M, [ ], A_iso, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<B> = B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, Bmask, [ ], B, [ ], [ ], desc, false) ;
        C1 = GB_mex_assign_alias_mask (C, B, desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = x where C is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (C, M, [ ], X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M> = x where C is dense
        %---------------------------------------

        C0 = GB_spec_assign (C, M, [ ], X, [ ], [ ], desc, true) ;
        C1 = GB_mex_assign  (C, M, [ ], X, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C<M,struct> = x
        %---------------------------------------

        C0 = GB_spec_assign (S, M, [ ], X, [ ], [ ], desc, true) ;
        C1 = GB_mex_assign  (S, M, [ ], X, [ ], [ ], desc) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % C = x
        %---------------------------------------

        C0 = GB_spec_assign (S, [ ], [ ], X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (S, [ ], [ ], X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1) ;

        %---------------------------------------
        % with accum operators
        %---------------------------------------

        for k2 = 1:length(binops)
            binop = binops {k2}  ;

            tol = [ ] ;
            switch (binop)
                case { 'pow', 'atan2', 'hypot', 'remainder' }
                    A.matrix = Amat2 ;
                    B.matrix = Bmat2 ;
                    C.matrix = Cmat2 ;
                    if (test_contains (type, 'single'))
                        tol = 1e-5 ;
                    elseif (test_contains (type, 'double'))
                        tol = 1e-12 ;
                    end
                otherwise
                    A.matrix = Amat ;
                    B.matrix = Bmat ;
                    C.matrix = Cmat ;
            end

            accum.opname = binop ;
            accum.optype = type ;

            try
                GB_spec_operator (accum) ;
            catch
                continue
            end

            if (GB_spec_is_positional (accum))
                continue ;
            end

            %---------------------------------------
            % C += A where A is dense
            %---------------------------------------

            C0 = GB_spec_assign (C, [ ], accum, A, [ ], [ ], [ ], false) ;
            C1 = GB_mex_assign  (C, [ ], accum, A, [ ], [ ], [ ]) ;
            GB_spec_compare (C0, C1, id, tol) ;

            %---------------------------------------
            % C += B where B is sparse
            %---------------------------------------

            C0 = GB_spec_assign (C, [ ], accum, B, [ ], [ ], [ ], false) ;
            C1 = GB_mex_assign  (C, [ ], accum, B, [ ], [ ], [ ]) ;
            GB_spec_compare (C0, C1, id, tol) ;

            %---------------------------------------
            % C += x
            %---------------------------------------

            C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], [ ], true) ;
            C1 = GB_mex_assign  (C, [ ], accum, X, [ ], [ ], [ ]) ;
            GB_spec_compare (C0, C1, id, tol) ;

            %---------------------------------------
            % C<replace> += x
            %---------------------------------------

            C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], drep, true) ;
            C1 = GB_mex_subassign  (C, [ ], accum, X, [ ], [ ], drep) ;
            GB_spec_compare (C0, C1, id, tol) ;

        end
    end
end

fprintf ('\ntest142: all tests passed\n') ;

