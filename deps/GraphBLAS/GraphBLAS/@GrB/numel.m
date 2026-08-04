function s = numel (G)
%NUMEL the maximum number of entries in a matrix.
% numel (G) is m*n for the m-by-n GraphBLAS matrix G.  If m, n, or m*n exceed
% flintmax (2^53), the result is returned as a vpa symbolic value, to avoid
% integer overflow.
%
% See also GrB/nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n] = gbmex_size (G) ;
s = m*n ;

if (m > flintmax || n > flintmax || s > flintmax)
    % use the VPA if available, for really huge matrices
    if (exist ('vpa', 'file'))
        s = vpa (vpa (m, 64) * vpa (n, 64), 128) ;
    end
end

