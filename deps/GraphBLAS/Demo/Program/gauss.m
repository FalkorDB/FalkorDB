% gauss.m: computes the same thing as gauss_demo.c

format short g
format compact
A = zeros (4,4) ;

for i = 0:3
    for j = 0:3
        A(i+1,j+1) = (i+1) + 1i * (2-j) ;
    end
end
A (1,1) = 0 ;

A
sum(A, 'all')

A = A^2
sum (A, 'all')

D = eye (4) ;
C = D .* (A*A.')

fprintf ('C=D*A\n') ;
D = diag (diag (A)) ;
C = D*A
fprintf ('C=A*D\n') ;
C = A*D

C = ones (4,4) * (1 - 2i) ;
C = C + A*A.'

B = ones (4,4) * (1 - 2i) ;
C = C + B*A

C = C + A*B

C = (1-2i) + A
C = A * (1-2i)
C = A.' * (1-2i)
C = (1-2i) * A.'

R = real (C)
R = real (C')

S = zeros (4,4) ;
for i = 0:3
    for j = 0:3
        S(i+1,j+1) = i - j ;
    end
end

C
R = real (C) + 1 + S
