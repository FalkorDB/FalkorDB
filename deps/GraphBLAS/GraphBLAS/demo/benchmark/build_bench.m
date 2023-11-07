
clear all
Prob = ssget ('GAP/GAP-twitter') ;
A = Prob.A ;
clear Prob

n = size (A,1) ;

tic
[I,J,X] = find (A) ;
toc

I64 = int64 (I) ;
J64 = int64 (J) ;

tic ; C = sparse (I, J, X, n, n) ; toc ; clear C
tic ; C = GrB.build (I, J, X, n, n) ; toc ; clear C
tic ; C = GrB.build (I64, J64, X, n, n) ; toc ; clear C

% build the transpose (to force a sort)
tic ; C = sparse (J, I, X, n, n) ; toc ; clear C
tic ; C = GrB.build (J, I, X, n, n) ; toc ; clear C
tic ; C = GrB.build (J64, I64, X, n, n) ; toc ; clear C

