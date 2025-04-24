% create the pushpull.mtx
clear all

A1 = sprand (1000, 1000, 0.05) ;
A1 = A1 + A1' ;

A2 = sprand (1000, 1000, 0.05) ;
A2 = A2 + A2' ;

A3 = sparse (1000, 1000) ;
A3 (1000,1) = 1 ;

e = ones (1000,1 ) ;
D = spdiags ([e e e], -1:1, 1000, 1000) ;

S = cell (4,4) ;
for i = 1:4
    for j = 1:4
        S {i,j} = sparse (1000, 1000) ;
    end
end

S {1,1} = A1 ;

S {1,2} = A3 ;
S {2,1} = A3' ;

S {2,2} = D ;

S {2,3} = A3 ;
S {3,2} = A3' ;

S {3,3} = A2 ;

S {3,4} = A3 ;
S {4,3} = A3' ;

S {4,4} = D ;

C = cell2mat (S) ;

mwrite (C, 'pushpull.mtx')
