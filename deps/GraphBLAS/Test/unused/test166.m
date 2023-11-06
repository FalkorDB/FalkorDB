function test166
%TEST166 GxB_select with a dense matrix

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
fprintf ('test166: ') ;

n = 10 ;
A.matrix = rand (n) ;
A.matrix (1,1) = 0 ;
A.pattern = true (n) ;
A.sparsity = 8 ;
Cin = sparse (n, n) ;

[~, ~, ~, ~, ~, select_ops] = GB_spec_opsall ;

thunk = 0 ;

for k = 1:length(select_ops)

    op = select_ops {k} ;
    fprintf ('%s ', op) ;
    if (mod (k, 4) == 0)
        fprintf ('\n') ;
    end

    % no mask, thunk = 0
    C1 = GB_spec_select (Cin, [], [], op, A, thunk, []) ;
    C2 = GB_mex_select  (Cin, [], [], op, A, thunk, [], 'test') ;
    GB_spec_compare (C1, C2) ;

end

fprintf ('resize\n') ;
C1 = GB_spec_resize (A, 5, 15) ;
C2 = GB_mex_resize  (A, 5, 15) ;
GB_spec_compare (C1, C2) ;
fprintf ('test166: all tests passed\n') ;

