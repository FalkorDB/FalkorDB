function test136
%TEST136 GxB_subassign, method 08, 09, 11

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test136: GxB_subassign, special cases\n') ;

rng ('default') ;

m = 1000 ;
n = 5 ;
am = 500 ;
an = n ;

C = sprand (m, n, 0.1) ;
I = randperm (m, am) ;

M = spones (sprand (am, an, 0.1)) ;
A = sprand (am, an, 0.1) ;
I0 = uint64 (I) - 1 ;

M (:,1) = 0 ;
M (1:2,1) = 1 ;
A (:,1) = sprand (am, 1, 0.8)  ;

A (:,2) = 0 ;
A (1:2,2) = 1 ;
M (:,2) = spones (sprand (am, 1, 0.8))  ;

% Method 08: C(I,J)<M> += A
C2 = GB_mex_subassign  (C, M, 'plus', A, I0, [ ], [ ]) ;
C1 = GB_spec_subassign (C, M, 'plus', A, I , [ ], [ ], false) ;
GB_spec_compare (C1, C2) ;

% create a Mask with explicit zero entries
[i j x] = find (M) ;
nz = length (x) ;
p = randperm (nz, floor(nz/2)) ;
x (p) = 0 ;
i = uint64 (i-1) ;
j = uint64 (j-1) ;
Mask = GB_mex_Matrix_build (i,j,x,am,an,[]) ;
Mask = Mask.matrix ;

% Method 09: C(I,J)<M,repl> = scalar
scalar = sparse (pi) ;
desc.outp = 'replace' ;
C2 = GB_mex_subassign  (C, Mask, [ ], scalar, I0, [ ], desc) ;
C1 = GB_spec_subassign (C, Mask, [ ], scalar, I , [ ], desc, true) ;
GB_spec_compare (C1, C2) ;

% Method 11: C(I,J)<M,repl> += scalar
scalar = sparse (pi) ;
desc.outp = 'replace' ;
C2 = GB_mex_subassign  (C, Mask, 'plus', scalar, I0, [ ], desc) ;
C1 = GB_spec_subassign (C, Mask, 'plus', scalar, I , [ ], desc, true) ;
GB_spec_compare (C1, C2) ;

% repeat method 02, subassignment with zombies:
% subref triggers case 4 with zombies
% no pending tuples
clear desc
d = [ ] ;
m = 3 ;
n = 8 ;
C = sparse (rand (m,n)) ;
C = tril (C,-1) + triu (C,1) ;

Work (1).A = sparse (m,2) ;
Work (1).I = [ ] ;
Work (1).J = [1 2] ;
Work (1).desc = d ;
Work (2).A = sparse (m,2) ;
Work (2).I = [ ] ;
Work (2).J = [2 3] ;
Work (2).desc = d ;
Work (3).A = sparse (m,2) ;
Work (3).I = [ ] ;
Work (3).J = [4 8] ;
Work (3).desc = d ;

Work2 (1).A = sparse (m,2) ;
Work2 (1).I = [ ] ;
Work2 (1).J = uint64 ([1 2]) - 1 ;
Work2 (1).desc = d ;
Work2 (2).A = sparse (m,2) ;
Work2 (2).I = [ ] ;
Work2 (2).J = uint64 ([2 3])  - 1;
Work2 (2).desc = d ;
Work2 (3).A = sparse (m,2) ;
Work2 (3).I = [ ] ;
Work2 (3).J = uint64 ([4 8]) - 1 ;
Work2 (3).desc = d ;

C1 = C ;
for k = 1:length (Work)
    C1 = GB_spec_subassign (C1, [ ], [ ], ...
        Work (k).A, Work (k).I, Work (k).J, Work (k).desc, false) ;
end

C2 = GB_mex_subassign (C, Work2) ;  % WORK_ASSIGN
GB_spec_compare (C1, C2) ;

C2 = GB_mex_subassign (C, Work2, [2 2]) ;   % WORK_ASSIGN
GB_spec_compare (C1, C2) ;

fprintf ('test136: all tests passed\n') ;
