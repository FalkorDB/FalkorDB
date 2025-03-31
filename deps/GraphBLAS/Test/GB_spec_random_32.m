function C = GB_spec_random_32 (A, x)
%GB_SPEC_RANDOM_32 select 32/64 bit format at random
%
% C = GB_spec_random_32 (A, x)
%
% A: a MATLAB matrix, or a struct with A.matrix
% x: a random scalar, either uniform or standard distribution
%
% C.p_is_32, C.j_is_32, and C.i_is_32 are set at random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 2)
    x = rand ;
end

if (isstruct (A))
    C = A ;
else
    C.matrix = A ;
end

is_csc = true ;
if (isfield (C, 'is_csc'))
    is_csc = C.is_csc ;
end

x = 1000 * abs (x) ;
x = x - fix (x) ;
x = 8 * x ;
x = floor (x) + 1 ;
C.p_is_32 = (bitand (x, 4) ~= 0) ;
C.j_is_32 = (bitand (x, 2) ~= 0) ;
C.i_is_32 = (bitand (x, 1) ~= 0) ;
if (nnz (C.matrix) > 2^31)
    C.p_is_32 = false ;
end
[m n] = size (C.matrix) ;
if (is_csc)
    if (n > 2^30)
        C.j_is_32 = false ;
    end
    if (m > 2^30)
        C.i_is_32 = false ;
    end
else
    if (n > 2^30)
        C.i_is_32 = false ;
    end
    if (m > 2^30)
        C.j_is_32 = false ;
    end
end


