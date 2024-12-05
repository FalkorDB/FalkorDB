function test154
%TEST154 test GrB_apply with scalar binding

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
ops = binops.all ;
types = types.all ;

fprintf ('\n--- testing apply with binaryop and scalar binding\n') ;
rng ('default') ;
% the right approach for apply_bind1st and apply_bind2nd
desc0.inp0 = 'tran' ;
desc1.inp1 = 'tran' ;
% shotgun approach for eWiseMult
desc.inp0 = 'tran' ;
desc.inp1 = 'tran' ;

n_operators = 0 ;
for k2 = 1:length(ops)
    mulop = ops {k2} ;
    fprintf ('\n%-10s ', mulop) ;

    for k1 = 1:length (types)
    type = types {k1} ;

    % create the op
    clear op
    op.opname = mulop ;
    op.optype = type ;

    try
        [oname ot ztype xtype ytype] = GB_spec_operator (op) ;
    catch
        continue ;
    end
    n_operators = n_operators + 1  ;

    switch (mulop)
        case { 'pow' }
            xlimits = [0, 5] ;
            ylimits = [0, 5] ;
        case { 'ldexp' }
            xlimits = [-5, 5] ;
            ylimits = [-5, 5] ;
        otherwise
            xlimits = [ ] ;
            ylimits = [ ] ;
    end

    if (test_contains (type, 'single'))
        tol = 1e-5 ;
    elseif (test_contains (type, 'double'))
        tol = 1e-12 ;
    else
        tol = 0 ;
    end

    fprintf ('.') ;

    for m = [1 4] % [ 1 10 ]% 100]
    for n = [1 4] % [1 10 ]% 100]
    for hi = [1 5] % [-1:2:5 ]
    for lo = [-1 0] % [-3:2:5 ]
    Amat = (hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5) ;
    Bmat = (hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5) ;
    xmat = (hi*sparse (rand(1))-lo) .* sparse (rand(1)) ;
    ymat = (hi*sparse (rand(1))-lo) .* sparse (rand(1)) ;
    Cmat = sparse (m, n) ;

    if (~isempty (xlimits))
        Amat = max (Amat, xlimits (1)) ;
        Amat = min (Amat, xlimits (2)) ;
        xmat = max (xmat, xlimits (1)) ;
        xmat = min (xmat, xlimits (2)) ;
    end
    if (xmat == 0)
        xmat = sparse (0.5) ;
    end

    if (~isempty (ylimits))
        Bmat = max (Bmat, ylimits (1)) ;
        Bmat = min (Bmat, ylimits (2)) ;
        ymat = max (ymat, ylimits (1)) ;
        ymat = min (ymat, ylimits (2)) ;
    end
    if (ymat == 0)
        ymat = sparse (0.5) ;
    end

    C.matrix = Cmat ;
    C.class = ztype ;

    CT.matrix = Cmat' ;
    CT.class = ztype ;

    A.matrix = Amat ;
    A.class = xtype ;

    B.matrix = Bmat ;
    B.class = ytype ;

    x.matrix = xmat ;
    x.class = xtype ;

    y.matrix = ymat ;
    y.class = ytype ;

    X.matrix = xmat .* spones (Bmat) ;
    X.class = xtype ;

    Y.matrix = ymat .* spones (Amat) ; 
    Y.class = ytype ;

    op_ewise_bind1st = op ;
    op_ewise_bind2nd = op ;
    if (isequal (op.opname, 'any'))
        op_ewise_bind1st.opname = 'first' ;
        op_ewise_bind2nd.opname = 'second' ;
    end

    C1 = GB_mex_apply1 (C, [ ], [ ], op, 0, x, B) ;
    C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op_ewise_bind1st, X, B, [ ]) ;
    GB_spec_compare (C1, C2, 0, tol) ;
    C1 = GB_mex_apply1 (C, [ ], [ ], op, 1, x, B) ;
    GB_spec_compare (C1, C2, 0, tol) ;

    C1 = GB_mex_apply1 (CT, [ ], [ ], op, 0, x, B, desc1) ;
    C2 = GB_spec_Matrix_eWiseMult (CT, [ ], [ ], op_ewise_bind1st, X, B, desc) ;
    GB_spec_compare (C1, C2, 0, tol) ;
    C1 = GB_mex_apply1 (CT, [ ], [ ], op, 1, x, B, desc1) ;
    GB_spec_compare (C1, C2, 0, tol) ;

    for csc = 0:1

        A.is_csc = csc ;
        C.is_csc = csc ;
        CT.is_csc = csc ;

        C1 = GB_mex_apply2 (C, [ ], [ ], op, 0, A, y) ;
        C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op_ewise_bind2nd, A, Y, [ ]) ;
        GB_spec_compare (C1, C2, 0, tol) ;
        C1 = GB_mex_apply2 (C, [ ], [ ], op, 1, A, y) ;
        GB_spec_compare (C1, C2, 0, tol) ;

        C1 = GB_mex_apply2 (CT, [ ], [ ], op, 0, A, y, desc0) ;
        C2 = GB_spec_Matrix_eWiseMult (CT, [ ], [ ], op_ewise_bind2nd, A, Y, desc) ;
        GB_spec_compare (C1, C2, 0, tol) ;
        C1 = GB_mex_apply2 (CT, [ ], [ ], op, 1, A, y, desc0) ;
        GB_spec_compare (C1, C2, 0, tol) ;
    end

    y.class = 'double' ;
    Y.class = 'double' ;

    C1 = GB_mex_apply2 (C, [ ], [ ], op, 0, A, y) ;
    C2 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op_ewise_bind2nd, A, Y, [ ]) ;
    GB_spec_compare (C1, C2, 0, tol) ;

    C1 = GB_mex_apply2 (C, [ ], [ ], op, 1, A, y) ;
    GB_spec_compare (C1, C2, 0, tol) ;

    C1 = GB_mex_apply2 (CT, [ ], [ ], op, 0, A, y, desc0) ;
    C2 = GB_spec_Matrix_eWiseMult (CT, [ ], [ ], op_ewise_bind2nd, A, Y, desc) ;
    GB_spec_compare (C1, C2, 0, tol) ;
    C1 = GB_mex_apply2 (CT, [ ], [ ], op, 1, A, y, desc0) ;
    GB_spec_compare (C1, C2, 0, tol) ;

end
end
end
end
end
end

fprintf ('\nNumber of built-in GraphBLAS operators: %d\n',  n_operators) ;
fprintf ('\ntest154: all tests passed\n') ;

