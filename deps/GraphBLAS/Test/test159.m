function test159
%TEST159 test dot and saxpy with positional ops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[binops, ~, ~, ~, ~, ~] = GB_spec_opsall ;
pos = binops.positional ;
pos {end+1} = 'times' ;
pos {end+1} = 'div' ;
pos {end+1} = 'first' ;
pos {end+1} = 'second' ;

n = 10 ;
A = GB_spec_random (n, n, 0.05, 256, 'int64') ;
B = GB_spec_random (n, n, 0.05, 256, 'int64') ;

dnn = struct ;
dtn = struct ('inp0', 'tran') ;
dnt = struct ('inp1', 'tran') ;
dtt = struct ('inp0', 'tran', 'inp1', 'tran') ;

Cin = sparse (n,n) ;

semiring.add = 'plus' ;

A.class = 'int32' ;
B.class = 'double' ;

for c = 1:4

    if (c == 1 || c == 2)
        dnn.axb = 'saxpy' ;
        dtn.axb = 'saxpy' ;
        dnt.axb = 'saxpy' ;
        dtt.axb = 'saxpy' ;
    else
        dnn.axb = 'dot' ;
        dtn.axb = 'dot' ;
        dnt.axb = 'dot' ;
        dtt.axb = 'dot' ;
    end

    if (c == 1 || c == 3)
        semiring.class = 'int64' ;
    else
        semiring.class = 'int32' ;
    end

    fprintf ('\ntypes: %s %s %s\n', A.class, B.class, semiring.class) ;

    for k = 1:length(pos)

        op = pos {k} ;
        fprintf ('op: %s\n', op) ;
        semiring.multiply = op ;

        % C = A*B
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dnn) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dnn) ;
        C3 = GB_mex_mxm_generic  (Cin, [ ], [ ], semiring, A, B, dnn) ;
        GB_spec_compare (C1, C2) ;
        GB_spec_compare (C1, C3) ;

        % C = A'*B
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dtn) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dtn) ;
        C3 = GB_mex_mxm_generic  (Cin, [ ], [ ], semiring, A, B, dtn) ;
        GB_spec_compare (C1, C2) ;
        GB_spec_compare (C1, C3) ;

        % C = B*B'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, B, A, dnt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, B, A, dnt) ;
        C3 = GB_mex_mxm_generic  (Cin, [ ], [ ], semiring, B, A, dnt) ;
        GB_spec_compare (C1, C2) ;
        GB_spec_compare (C1, C3) ;

        % C = A'*B'
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dtt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dtt) ;
        C3 = GB_mex_mxm_generic  (Cin, [ ], [ ], semiring, A, B, dtt) ;
        GB_spec_compare (C1, C2) ;
        GB_spec_compare (C1, C3) ;

        % C = B'*A
        C1 = GB_spec_mxm (Cin, [ ], [ ], semiring, B, A, dtt) ;
        C2 = GB_mex_mxm  (Cin, [ ], [ ], semiring, B, A, dtt) ;
        C3 = GB_mex_mxm_generic (Cin, [ ], [ ], semiring, B, A, dtt) ;
        GB_spec_compare (C1, C2) ;
        GB_spec_compare (C1, C3) ;

    end

end
fprintf ('\ntest159: all tests passed\n') ;

