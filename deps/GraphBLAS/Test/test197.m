function test197
%TEST197 test large sparse split

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
GB_mex_burble (0) ;
n = 2e6 ;
d = 1000 / n^2 ;
A.matrix = sprand (n, n, d) ; A.sparsity = 1 ;
A.matrix (:,1) = 1 ;

k = 1e6 ;
ns = [k k] ;

C2 {1,1} = A.matrix (1:k, 1:k) ;
C2 {1,2} = A.matrix (1:k, (k+1):n) ;
C2 {2,1} = A.matrix ((k+1):n, 1:k) ;
C2 {2,2} = A.matrix ((k+1):n, (k+1):n) ;

C1 = GB_mex_split (A, ns, ns) ;
for i = 1:length(ns)
    for j = 1:length(ns)
        C = C1 {i,j} ;
        assert (isequal (C.matrix, C2 {i,j})) ;
    end
end

fprintf ('test197: all tests passed\n') ;

