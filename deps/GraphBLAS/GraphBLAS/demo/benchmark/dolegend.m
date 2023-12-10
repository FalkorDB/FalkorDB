function dolegend

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('S: large input sparse matrix (m-by-n)\n') ;
fprintf ('x: dense vector (1-by-m or n-by-1)\n') ;
fprintf ('F: dense matrix (4-by-m or n-by-4)\n') ;
fprintf ('L: 8-by-m sparse matrix, about 1000 entries\n') ;
fprintf ('R: n-by-8 sparse matrix, about 1000 entries\n') ;
fprintf ('B: m-by-n sparse matrix, about nnz(S)/10 entries\n') ;
fprintf ('p,q: random permutation vectors\n') ;
