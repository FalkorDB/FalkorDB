function test230
%TEST230 test GrB_apply with idxunop

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~, idxunops] = GB_spec_opsall ;
ops = idxunops ;
types = types.all ;

fprintf ('\n--- testing apply with idxunops\n') ;
rng ('default') ;

defaults = [ ] ;
desc.inp0 = 'tran' ;

n_operators = 0 ;
for k2 = 1:length(ops)
    opname = ops {k2} ;
    fprintf ('\n%-10s ', opname) ;

    for k1 = 1:length (types)
    type = types {k1} ;

    % create the op
    clear op
    op.opname = opname ;
    op.optype = type ;

    [is_idxunop, ztype] = GB_spec_is_idxunop (opname, type) ;
    if (~is_idxunop)
        continue ;
    end

    fprintf ('.') ;
    n_operators = n_operators + 1 ;

    for m = [1 4] % [ 1 10 ]% 100]
    for n = [1 4] % [1 10 ]% 100]
    for hi = [1 5] % [-1:2:5 ]
    for lo = [-1 0] % [-3:2:5 ]
    Amat = (hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5) ;
    Bmat = (hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5) ;
    Cmat = sparse (m, n) ;

    C.matrix = Cmat ;
    C.class = ztype ;

    CT.matrix = Cmat' ;
    CT.class = ztype ;

    A.matrix = Amat ;
    A.class = type ;

    for ythunk = -3:3
    y.matrix = ythunk ;
    y.class = type ;

    for how = 0:1
    for csc = 0:1

    A.is_csc = csc ;
    C.is_csc = csc ;
    CT.is_csc = csc ;

    C1 = GB_mex_apply_idxunop (C, [ ], [ ], op, how, A,           y, defaults) ;
    C2 = GB_spec_apply        (C, [ ], [ ], op,      A, defaults, y) ;
    GB_spec_compare (C1, C2) ;

    C1 = GB_mex_apply_idxunop (CT, [ ], [ ], op, how, A,           y, desc) ;
    C2 = GB_spec_apply        (CT, [ ], [ ], op,      A, desc    , y) ;
    GB_spec_compare (C1, C2) ;

    end

end
end
end
end
end
end
end
end

fprintf ('\nNumber of built-in GraphBLAS idxunops: %d\n',  n_operators) ;
fprintf ('\ntest230: all tests passed\n') ;

