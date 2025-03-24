function test125
%TEST125 test GrB_mxm: row and column scaling
% all built-in semirings, no typecast, no mask

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

fprintf ('-------------- GrB_mxm on all semirings (row,col scale)\n') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

rng ('default') ;

n = 10 ;

n_semirings = 0 ;
A = GB_spec_random (n,n,0.3,100,'none') ;
clear B
B1matrix = spdiags (3 * rand (n,1), 0, n, n) ;
B.matrix = B1matrix ;
B.class = 'none' ;
B.pattern = logical (spones (B1matrix)) ;

C = GB_spec_random (n,n,0.3,100,'none') ;
M = spones (sprandn (n, n, 0.3)) ;

for k1 = 1:length(mult_ops)
    mulop = mult_ops {k1} ;
    fprintf ('\n%-10s ', mulop) ;
    nmult_semirings = 0 ;

    for k2 = 1:length(add_ops)
        addop = add_ops {k2} ;
        fprintf ('.') ;

        for k3 = 1:length (types)
            type = types {k3} ;

            semiring.multiply = mulop ;
            semiring.add = addop ;
            semiring.class = type ;

            % semiring

            % create the semiring.  some are not valid because the
            % or,and,xor monoids can only be used when z is boolean for
            % z=mult(x,y).
            try
                [mult_op add_op id] = GB_spec_semiring (semiring) ;
                [mult_opname mult_optype ztype xtype ytype] = ...
                    GB_spec_operator (mult_op) ;
                [ add_opname  add_optype] = GB_spec_operator (add_op) ;
                identity = GB_spec_identity (semiring.add, add_optype) ;
            catch
                continue
            end

            n_semirings = n_semirings + 1 ;
            nmult_semirings = nmult_semirings + 1 ;
            A.class = type ;
            B.class = type ;
            C.class = type ;

            % C = A*B
            C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, dnn);
            C0 = GB_spec_mxm (C, [ ], [ ], semiring, A, B, dnn);
            GB_spec_compare (C0, C1, identity) ;

            % C = B*A
            C1 = GB_mex_mxm  (C, [ ], [ ], semiring, B, A, dnn);
            C0 = GB_spec_mxm (C, [ ], [ ], semiring, B, A, dnn);
            GB_spec_compare (C0, C1, identity) ;

        end
    end
    fprintf (' %4d', nmult_semirings) ;
end

fprintf ('\nsemirings tested: %d\n', n_semirings) ;
fprintf ('\ntest125: all tests passed\n') ;

