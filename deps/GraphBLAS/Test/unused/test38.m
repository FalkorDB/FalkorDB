function test38
%TEST38 test GrB_transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n ----------- testing GB_mex_transpose on SuiteSparse matrices\n') ;

% test GB_mex_transpose on lots of matrices from the
% SuiteSparse matrix collection

index = ssget ;
[~, f] = sort (index.nnz) ;
nmat = length (f) ;

for k = 1:500
    fprintf ('.') ;
    id = f (k) ;
    Prob = ssget (id) ;
    A = Prob.A ;
    if (~isreal (A))
        A = real (A) ;
    end
    [m n] = size (A) ;
    % tic
    C = A' ;
    % toc
    % tic
    S = GB_mex_transpose (sparse (n,m), [ ], [ ], A) ;
    % toc
    assert (isequal (C,S.matrix)) ;
end

fprintf ('\ntest38: all tests passed\n') ;

