function test175
%TEST175 test GrB_assign with accum

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
types = types.all ;

fprintf ('test175 ------------ GrB_assign with accum\n') ;

m = 10 ;
n = 14 ;

rng ('default') ;

M1.matrix = logical (spones (sprand (m, n, 0.5))) ;
M1.sparsity = 2 ; % sparse

M2.matrix = logical (spones (sprand (m, n, 0.5))) ;
M2.sparsity = 4 ; % sparse

Amat2 = sparse (2 * rand (m,n)) ;
Bmat2 = sparse (2 * sprand (m,n, 0.5)) ;
Cmat2 = sparse (2 * rand (m,n)) ;

Amat = 5000 * Amat2 ;
B1mat = 5000 * Bmat2 ;
Cmat = 5000 * Cmat2 ;

B2mat = rand (m,n) ;

Smat = sparse (m,n) ;
Xmat = sparse (pi) ;
desc.mask = 'structural' ;
drep.outp = 'replace' ;

for k1 = 1:length (types)
    type = types {k1}  ;
    fprintf ('%s, ', type) ;

    clear A B1 B2 C S X
    A.matrix = Amat   ; A.class = 'see below' ;
    B1.matrix = B1mat ; B1.class = 'see below' ;
    B2.matrix = B2mat ; B2.class = 'see below' ;
    C.matrix = Cmat   ; C.class = 'see below' ;
    S.matrix = Smat   ; S.class = 'see below' ;
    X.matrix = Xmat   ; X.class = 'see below' ;
    A.class = type ;

    for k3 = 1:3

    if (k3 == 1)
        X.class = type ;
        B1.class = type ;
        B2.class = type ;
        C.class = 'logical' ;
        S.class = 'logical' ;
    elseif (k3 == 2)
        X.class = type ;
        B1.class = type ;
        B2.class = type ;
        C.class = type ;
        S.class = type ;
    else
        X.class = 'int8' ;
        B1.class = 'int8' ;
        B2.class = 'int8' ;
        C.class = type ;
        S.class = type ;
    end

    for k4 = 1:7

        if (k4 == 1)
            M = M1 ;
            B = B1 ;
            C.sparsity = 2 ;
            B.sparsity = 2 ;
        elseif (k4 == 2)
            M = M2 ;
            B = B1 ;
            C.sparsity = 2 ;
            B.sparsity = 2 ;
        elseif (k4 == 3)
            M = M1 ;
            B = B1 ;
            C.sparsity = 4 ;
            B.sparsity = 2 ;
        elseif (k4 == 4)
            M = M2 ;
            B = B1 ;
            C.sparsity = 4 ;
            B.sparsity = 2 ;
        elseif (k4 == 5)
            M = M2 ;
            B = B1 ;
            C.sparsity = 4 ;
            B.sparsity = 4 ;
        elseif (k4 == 6)
            M = M2 ;
            B = B2 ;
            C.sparsity = 4 ;
            B.sparsity = 4 ;
        elseif (k4 == 7)
            M = M2 ;
            B = B2 ;
            C.sparsity = 2 ;
            B.sparsity = 4 ;
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

    end

    %---------------------------------------
    % C = x
    %---------------------------------------

    C0 = GB_spec_assign (S, [ ], [ ], X, [ ], [ ], [ ], true) ;
    C1 = GB_mex_assign  (S, [ ], [ ], X, [ ], [ ], [ ]) ;
    GB_spec_compare (C0, C1) ;

    %---------------------------------------
    % with accum operators
    %---------------------------------------

    clear A B C

    for k2 = 1:length(binops)
        binop = binops {k2}  ;

        tol = 0 ;
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
                B.matrix = B1mat ;
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
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % C += B where B is sparse
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, B, [ ], [ ], [ ], false) ;
        C1 = GB_mex_assign  (C, [ ], accum, B, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % C += x
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], [ ], true) ;
        C1 = GB_mex_assign  (C, [ ], accum, X, [ ], [ ], [ ]) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % C<replace> += x
        %---------------------------------------

        C0 = GB_spec_assign (C, [ ], accum, X, [ ], [ ], drep, true) ;
        C1 = GB_mex_subassign  (C, [ ], accum, X, [ ], [ ], drep) ;
        GB_spec_compare (C0, C1, 0, tol) ;

    end
end
end

fprintf ('\ntest175: all tests passed\n') ;

