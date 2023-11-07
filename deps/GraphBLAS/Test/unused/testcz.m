threads {1} = [4 1] ;
t = threads ;

% user-defined complex type
% logstat ('testc1(0)',t) ;  % test ops
logstat ('testc2(0,0)',t) ;  % A'*B, A+B, A*B
% logstat ('testc3(0)',t) ;  % extract column, extract submatrix
logstat ('testc4(0)',t) ;  % extractElement, setElement
% logstat ('testc5(0)',t) ;  % subref
% logstat ('testc6(0)',t) ;  % apply
logstat ('testc7(0)',t) ;  % assign
% logstat ('testc8(0)',t) ;  % eWise
% logstat ('testc9(0)',t) ;  % extractTuples
% logstat ('testca(0)',t) ;  % mxm, mxv, vxm
% logstat ('testcb(0)',t) ;  % reduce
% logstat ('testcc(0)',t) ;  % transpose

% builtin complex type: GxB_FC64
% logstat ('testc1(1)',t) ;  % test ops
% logstat ('testc2(0,1)',t) ;  % A'*B, A+B, A*B
% logstat ('testc3(1)',t) ;  % extract column, extract submatrix
% logstat ('testc4(1)',t) ;  % extractElement, setElement
% logstat ('testc5(1)',t) ;  % subref
% logstat ('testc6(1)',t) ;  % apply
logstat ('testc7(1)',t) ;  % assign
% logstat ('testc8(1)',t) ;  % eWise
% logstat ('testc9(1)',t) ;  % extractTuples
% logstat ('testca(1)',t) ;  % mxm, mxv, vxm
% logstat ('testcb(1)',t) ;  % reduce
logstat ('testcc(1)',t) ;  % transpose

