function test75b
%TEST75B GrB_mxm and GrB_vxm on all semirings (shorter test than test75)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test75b: test mxm and vxm\n') ;
[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

rng ('default') ;

m = 200 ;
n = 5 ;
A_sparse = sprandn (m, n, 0.1) ;
A_sparse (:,3) = 0 ;
A_sparse (2,3) = 1.7 ;
A_sparse (18,3) = 2.2 ;
A_sparse (:,1:2) = sparse (rand (m,2)) ;
A_sparse (1,1) = 0;
A_sparse (18,1) = 0;
A_sparse (:,5) = 0 ;
A_sparse (1,5) = 11 ;
A_sparse (2,5) = 23 ;
A_sparse (18,5) = 33 ;

B_sparse = sprandn (m, n, 0.1) ;
B_sparse (:,1) = 0 ;
B_sparse (1,1) = 3 ;
B_sparse (18,1) = 2 ;
B_sparse (:,[2 n]) = sparse (rand (m,2)) ;
B_sparse (3,2) = 0 ;
B_sparse (18,2) = 0 ;
A_sparse (:,3) = 0 ;
B_sparse (2,1) = 7 ;
B_sparse (18,1) = 8 ;
B_sparse (19,1) = 9 ;

x_sparse = sparse (rand (m,1)) ;
x_sparse (99) = 0 ;

y_sparse = sparse (zeros (m,1)) ;
y_sparse (99) = 1 ;

A.matrix = A_sparse ;
A.class = 'see below' ;
A.pattern = logical (spones (A_sparse)) ;

B.matrix = B_sparse ;
B.class = 'see below' ;
B.pattern = logical (spones (B_sparse)) ;

X.matrix = x_sparse ;
X.class = 'see below' ;
X.pattern = logical (spones (x_sparse)) ;

Y.matrix = y_sparse ;
Y.class = 'see below' ;
Y.pattern = logical (spones (y_sparse)) ;

Cin = sparse (n, n) ;

Din = 10 * sparse (rand (n, n)) ;
D.matrix = Din ;
D.class = 'see below' ;
D.pattern = true (n,n) ;

Xin = sparse (n, 1) ;

Mask = sparse (ones (n,n)) ;
mask = sparse (ones (n,1)) ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dtn_dot   = struct ( 'inp0', 'tran', 'axb', 'dot' ) ;
dtn_saxpy = struct ( 'inp0', 'tran', 'axb', 'saxpy' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

% testing just 154 semirings
semirings = {
    { 'min', 'first', 'logical' },
    { 'min', 'first', 'int8' },
    { 'min', 'first', 'int16' },
    { 'min', 'first', 'uint8' },
    { 'min', 'first', 'uint16' },
    { 'min', 'first', 'uint32' },
    { 'min', 'first', 'uint64' },
    { 'min', 'first', 'single' },
    { 'min', 'first', 'double' },
    { 'plus', 'first', 'single complex' },
    { 'plus', 'first', 'double complex' },
    { 'min', 'second', 'logical' },
    { 'min', 'second', 'int8' },
    { 'min', 'second', 'int16' },
    { 'min', 'second', 'uint8' },
    { 'min', 'second', 'uint16' },
    { 'min', 'second', 'uint32' },
    { 'min', 'second', 'uint64' },
    { 'min', 'second', 'single' },
    { 'min', 'second', 'double' },
    { 'plus', 'second', 'single complex' },
    { 'plus', 'second', 'double complex' },
    { 'min', 'minus', 'int8' },
    { 'min', 'minus', 'int16' },
    { 'min', 'minus', 'int32' },
    { 'min', 'minus', 'int64' },
    { 'min', 'minus', 'uint8' },
    { 'min', 'minus', 'uint16' },
    { 'min', 'minus', 'uint32' },
    { 'min', 'minus', 'uint64' },
    { 'min', 'minus', 'single' },
    { 'min', 'minus', 'double' },
    { 'plus', 'minus', 'single complex' },
    { 'plus', 'minus', 'double complex' },
    { 'min', 'rminus', 'int8' },
    { 'min', 'rminus', 'int16' },
    { 'min', 'rminus', 'int32' },
    { 'min', 'rminus', 'int64' },
    { 'min', 'rminus', 'uint8' },
    { 'min', 'rminus', 'uint16' },
    { 'min', 'rminus', 'uint32' },
    { 'min', 'rminus', 'uint64' },
    { 'min', 'rminus', 'single' },
    { 'min', 'rminus', 'double' },
    { 'plus', 'rminus', 'single complex' },
    { 'plus', 'rminus', 'double complex' },
    { 'min', 'div', 'int8' },
    { 'min', 'div', 'int16' },
    { 'min', 'div', 'uint8' },
    { 'min', 'div', 'uint16' },
    { 'min', 'div', 'uint32' },
    { 'min', 'div', 'uint64' },
    { 'min', 'div', 'single' },
    { 'plus', 'div', 'single complex' },
    { 'plus', 'div', 'double complex' },
    { 'min', 'rdiv', 'int8' },
    { 'min', 'rdiv', 'int16' },
    { 'min', 'rdiv', 'int32' },
    { 'min', 'rdiv', 'int64' },
    { 'min', 'rdiv', 'uint8' },
    { 'min', 'rdiv', 'uint16' },
    { 'min', 'rdiv', 'uint32' },
    { 'min', 'rdiv', 'uint64' },
    { 'min', 'rdiv', 'single' },
    { 'min', 'rdiv', 'double' },
    { 'plus', 'rdiv', 'single complex' },
    { 'plus', 'rdiv', 'double complex' },
    { 'min', 'pow', 'logical' },
    { 'min', 'isgt', 'logical' },
    { 'min', 'isgt', 'int8' },
    { 'min', 'isgt', 'int16' },
    { 'min', 'isgt', 'int32' },
    { 'min', 'isgt', 'int64' },
    { 'min', 'isgt', 'uint8' },
    { 'min', 'isgt', 'uint16' },
    { 'min', 'isgt', 'uint32' },
    { 'min', 'isgt', 'uint64' },
    { 'min', 'isgt', 'single' },
    { 'min', 'isgt', 'double' },
    { 'min', 'islt', 'logical' },
    { 'min', 'islt', 'int8' },
    { 'min', 'islt', 'int16' },
    { 'min', 'islt', 'int32' },
    { 'min', 'islt', 'int64' },
    { 'min', 'islt', 'uint8' },
    { 'min', 'islt', 'uint16' },
    { 'min', 'islt', 'uint32' },
    { 'min', 'islt', 'uint64' },
    { 'min', 'islt', 'single' },
    { 'min', 'islt', 'double' },
    { 'min', 'isge', 'int8' },
    { 'min', 'isge', 'int16' },
    { 'min', 'isge', 'int32' },
    { 'min', 'isge', 'int64' },
    { 'min', 'isge', 'uint8' },
    { 'min', 'isge', 'uint16' },
    { 'min', 'isge', 'uint32' },
    { 'min', 'isge', 'uint64' },
    { 'min', 'isge', 'single' },
    { 'min', 'isge', 'double' },
    { 'min', 'isle', 'logical' },
    { 'min', 'isle', 'int8' },
    { 'min', 'isle', 'int16' },
    { 'min', 'isle', 'int32' },
    { 'min', 'isle', 'int64' },
    { 'min', 'isle', 'uint8' },
    { 'min', 'isle', 'uint16' },
    { 'min', 'isle', 'uint32' },
    { 'min', 'isle', 'uint64' },
    { 'min', 'isle', 'single' },
    { 'min', 'isle', 'double' },
    { 'min', 'gt', 'int8' },
    { 'min', 'gt', 'int16' },
    { 'min', 'gt', 'int32' },
    { 'min', 'gt', 'int64' },
    { 'min', 'gt', 'uint8' },
    { 'min', 'gt', 'uint16' },
    { 'min', 'gt', 'uint32' },
    { 'min', 'gt', 'uint64' },
    { 'min', 'gt', 'single' },
    { 'min', 'gt', 'double' },
    { 'min', 'lt', 'int8' },
    { 'min', 'lt', 'int16' },
    { 'min', 'lt', 'int32' },
    { 'min', 'lt', 'int64' },
    { 'min', 'lt', 'uint8' },
    { 'min', 'lt', 'uint16' },
    { 'min', 'lt', 'uint32' },
    { 'min', 'lt', 'uint64' },
    { 'min', 'lt', 'single' },
    { 'min', 'lt', 'double' },
    { 'min', 'ge', 'int8' },
    { 'min', 'ge', 'int16' },
    { 'min', 'ge', 'int32' },
    { 'min', 'ge', 'int64' },
    { 'min', 'ge', 'uint8' },
    { 'min', 'ge', 'uint16' },
    { 'min', 'ge', 'uint32' },
    { 'min', 'ge', 'uint64' },
    { 'min', 'ge', 'single' },
    { 'min', 'ge', 'double' },
    { 'min', 'le', 'int8' },
    { 'min', 'le', 'int16' },
    { 'min', 'le', 'int32' },
    { 'min', 'le', 'int64' },
    { 'min', 'le', 'uint8' },
    { 'min', 'le', 'uint16' },
    { 'min', 'le', 'uint32' },
    { 'min', 'le', 'uint64' },
    { 'min', 'le', 'single' },
    { 'min', 'le', 'double' },
    { 'min', 'firsti', 'int32' },
    { 'min', 'firsti', 'int64' },
    { 'min', 'firsti1', 'int32' }} ;

fprintf ('\n') ;

for k1 = 1:length(semirings)
    sr = semirings {k1} ;
    addop = sr {1} ;
    mulop = sr {2} ;
    type = sr {3} ;
    GB_mex_finalize ;

    semiring.multiply = mulop ;
    semiring.add = addop ;
    semiring.class = type ;

    % create the semiring.  some are not valid because the or,and,xor,eq
    % monoids can only be used when z is boolean with z=mult(x,y).
    try
        [mult_op add_op id] = GB_spec_semiring (semiring) ;
        [mult_opname mult_optype ztype xtype ytype] = ...
            GB_spec_operator (mult_op) ;
        [ add_opname  add_optype] = GB_spec_operator (add_op) ;
        identity = GB_spec_identity (semiring.add, add_optype) ;
    catch
        continue
    end

    A.class = type ;
    B.class = type ;
    X.class = type ;
    Y.class = type ;
    D.class = add_op.optype ;

    % C += A'*B, C dense, typecasting of C
    C1 = GB_mex_mxm  (Din, [ ], add_op, semiring, A, B, dtn_dot) ;
    C2 = GB_spec_mxm (Din, [ ], add_op, semiring, A, B, dtn) ;
    GB_spec_compare (C1, C2, id) ;

    % C += A'*B, C sparse, no typecasting of C
    C1 = GB_mex_mxm  (D, [ ], add_op, semiring, A, B, dtn_dot) ;
    C2 = GB_spec_mxm (D, [ ], add_op, semiring, A, B, dtn) ;
    GB_spec_compare (C1, C2, id) ;

    % X = u*A, with mask
    C1 = GB_mex_vxm  (Xin, mask, [ ], semiring, X, A, [ ]) ;
    C2 = GB_spec_vxm (Xin, mask, [ ], semiring, X, A, [ ]) ;
    GB_spec_compare (C1, C2, id) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            fprintf ('[%s.%s.%s %d %d]\n', addop, mulop, type, d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end

end

fprintf ('\ntest75b: all tests passed\n') ;


