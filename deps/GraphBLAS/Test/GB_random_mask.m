function Mask = GB_random_mask (m, n, d, M_is_csc, M_is_hyper)
% Mask = GB_random_mask (m, n, d, M_is_csc, M_is_hyper)
%
% Construct a random matrix, either as a matrix or a struct
% With 3 arguments, Mask is a sparse logical matrix.
% With 4, Mask is a struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

M = (sprand (m, n, d) ~= 0) ;

if (nargin < 4)
    Mask = M ;
else
    Mask.matrix   = M ;
    Mask.is_csc   = M_is_csc ;
    Mask.is_hyper = M_is_hyper ;
end

