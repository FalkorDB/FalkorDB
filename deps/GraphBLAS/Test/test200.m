function test200
%TEST200 test iso full matrix multiply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% GB_mex_burble (1) ;
n = 10 ;
A.matrix = pi * sparse (ones (n)) ;
A.sparsity = 8 ;    % full
A.iso = true ;
B.matrix = 3 * sparse (ones (n)) ;
B.sparsity = 8 ;    % full
B.iso = true ;
S.matrix = 4 * sparse (ones (n)) ;
S.sparsity = 8 ;    % full
S.iso = true ;
S.class = 'single' ;
M.matrix = spones (sprand (n, n, 0.5)) ;

fprintf ('.') ;
Cin = sparse (n,n) ;
dtn = struct ('inp0', 'tran') ;

% using the plus_times semiring:
semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

% C = A*B
C1 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, [ ]) ;
C2 = (A.matrix * B.matrix) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

% C<M> = A*B
C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;
C2 = (A.matrix * B.matrix) .* M.matrix ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

% C = A'*B
C1 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, dtn) ;
C2 = (A.matrix' * B.matrix) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

% C<M> = A'*B
C1 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, dtn) ;
C2 = (A.matrix' * B.matrix) .* M.matrix ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

% using other semirings

for op = { 'plus', 'pair', 'first', 'second' }  % pair == oneb
    for add = { 'plus', 'times', 'max', 'any' }

        fprintf ('.') ;
        semiring.add = add {1} ;
        semiring.multiply = op {1} ;
        semiring.class = 'double' ;

        % C = A*B
        C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, [ ]) ;
        C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % C = A*S
        C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, S, [ ]) ;
        C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, S, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % C = S*B
        C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, S, B, [ ]) ;
        C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, S, B, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % C<M> = A*B
        C1 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, [ ]) ;
        C2 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % C = A'*B
        C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, A, B, dtn) ;
        C2 = GB_spec_mxm (Cin, [ ], [ ], semiring, A, B, dtn) ;
        GB_spec_compare (C1, C2) ;

        % C<M> = A'*B
        C1 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, dtn) ;
        C2 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, dtn) ;
        GB_spec_compare (C1, C2) ;
    end
end

% GB_mex_burble (0) ;
fprintf ('\ntest200: all tests passed\n') ;

