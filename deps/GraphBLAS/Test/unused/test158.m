function test158
%TEST158 test colscale (A*D) and rowscale (D*B) with positional ops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[binops, ~, ~, ~, ~, ~] = GB_spec_opsall ;
pos = binops.positional ;
pos {end+1} = 'times' ;
pos {end+1} = 'div' ;

n = 30 ;
A = GB_spec_random (n, n, 0.05, 256, 'int64') ;
D.matrix = speye (n) ;
D.class = 'int64' ;
D.pattern = logical (spones (D.matrix)) ;

dnn = struct ;
dtn = struct ('inp0', 'tran') ;
dnt = struct ('inp1', 'tran') ;
dtt = struct ('inp0', 'tran', 'inp1', 'tran') ;

Cin = sparse (n,n) ;

semiring.add = 'plus' ;
semiring.class = 'int64' ;

for c = 1:3

    if (c == 1)
        A.class = 'int32' ;
        D.class = 'int32' ;
        semiring.class = 'int32' ;
    elseif (c == 2)
        A.class = 'int64' ;
        D.class = 'int64' ;
        semiring.class = 'int64' ;
    else
        A.class = 'int32' ;
        D.class = 'int64' ;
        semiring.class = 'int64' ;
    end

    fprintf ('\ntypes: %s %s %s\n', D.class, A.class, semiring.class) ;

    for k = 1:length(pos)

        op = pos {k} ;
        fprintf ('op: %s\n', op) ;
        semiring.multiply = op ;

        % colscale: C = A*D
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, D, [ ]) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, D, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % rowscale: C = D*A
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, D, A, [ ]) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, D, A, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % colscale: C = A'*D
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, D, dtn) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, D, dtn) ;
        GB_spec_compare (C1, C2) ;

        % colscale: C = D'*A
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, D, A, dtn) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, D, A, dtn) ;
        GB_spec_compare (C1, C2) ;

        % rowscale: C = D*A'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, D, A, dnt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, D, A, dnt) ;
        GB_spec_compare (C1, C2) ;

        % rowscale: C = A*D'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, D, dnt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, D, dnt) ;
        GB_spec_compare (C1, C2) ;

        % colscale: C = A'*D'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, D, dtt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, D, dtt) ;
        GB_spec_compare (C1, C2) ;

        % rowscale: C = D'*B'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, D, A, dtt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, D, A, dtt) ;
        GB_spec_compare (C1, C2) ;

    end

end

fprintf ('\ntest158: all tests passed\n') ;

