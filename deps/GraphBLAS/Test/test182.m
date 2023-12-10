function test182
%TEST182 test for internal wait that changes w from sparse/hyper to bitmap/full
%
% This test triggers the C<M>=A assignment where C starts out as sparse with
% has many pending tuples, and is converted to bitmap just before the
% assignment.  In this case, C is the vector w.  If w_sparsity is 15 and 'wait'
% is false, then it starts the w<v>=sum(A) reduction with many pending tuples,
% and converts w from sparse/hyper with many pending tuples into a bitmap
% vector.  The outputs w, v, and A should be the same, regardless of the input
% parameter s.
%
% The internal condition is triggered if wait is false, and w_sparsity
% is 5, 6, 7, 13, 14, or 15:
%
%   5:  4+1         bitmap (4) or hypersparse (1)
%   6:  4+2         bitmap (4) or sparse (2)
%   7:  4+2+1       bitmap (4), sparse (2), or hypersparse (1)
%  13:  8+4+1       full (8), bitmap (4) or hypersparse (1)
%  14:  8+4+2       full (8), bitmap (4) or sparse (2)
%  15:  8+4+2+1     full (8), bitmap (4), sparse (2), or hypersparse (1)
%
% That is, the sparsity control for w allows it to change from sparse/hyper
% (with pending updates) to/from bitmap.  If 'wait' is true, then GB_mex_gabor
% does an explicit GrB_Vector_wait on w before the w<v>=sum(A) reduction, so
% w is converted to bitmap before the assignment, not during, and the internal
% condition is not triggered.
%
% s is an optional vector of length 4, containing 4 parameters:
% s = [wait, w_sparsity, v_sparsity, A_sparsity] ;
% with wait 0 or 1, and the sparsity controls in range 1 to 15.
%
% The sparsity control for A and v has no effect on this condition.
%
% This test case comes from Gabor Szarnyas and Marton Elekes.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

I  = [ 1, 2, 4, 5, 7, 11, 12, 13, 15, 18, 19, 20, 27, 32, 33, ...
    35, 37, 41, 46, 50, 52, 53, 55, 57, 58, 61, 62, 63, 65, 66, 69, 70, 72, ...
    73, 74, 75, 78, 79, 81, 84, 86, 87, 90, 91, 94, 96, 97, 98, 99, 100, ...
    101, 102, 103, 104, 105, 107, 108, 109, 110, 115, 116, 117, 118, 120, ...
    123, 129, 131, 132, 133, 134, 136, 140, 145, 146, 149, 152, 153, 154, ...
    156, 158, 159, 160, 161, 163, 164, 165, 166, 168, 169, 172, 176, 177, ...
    181, 184, 186, 187, 189, 191, 193, 194, 195, 197, 200, 201, 202, 203, ...
    204, 205, 208, 209, 210, 211, 216, 217, 218, 219, 224, 225, 229, 230, ...
    232, 235, 236, 238, 239, 242, 243 ] ;

n = 1000 ;
Aok = sparse (I+1, I+1, I, n, n) ;
wok = diag (Aok) ;

for wait = 0:1
    for w_sparsity = 1:15
        fprintf ('.') ;
        for v_sparsity = [2 4 15] % 1:15
            for A_sparsity = [2 4 15] % 1:15
                s = [wait, w_sparsity, v_sparsity, A_sparsity] ;
                [w, v, A] = GB_mex_gabor (s) ;
                assert (isequal (Aok, A.matrix)) ;
                assert (isequal (wok, w.matrix)) ;
                assert (isequal (wok, v.matrix)) ;
            end
        end
    end
end

fprintf ('\ntest182: all tests passed\n') ;

