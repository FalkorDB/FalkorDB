function I = subsindex (G_arg)
%SUBSINDEX subscript index from a GraphBLAS matrix.
% I = subsindex (G) is an overloaded method used when the GraphBLAS GrB or GhB
% matrix G is used to index into a non-GraphBLAS matrix A, for A(G).
%
% See also GrB/subsref, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% On input, G_arg must contain integers in the range 1 to prod (size (A))-1.
% The dimensions of A are not provided to subsindex.

if (gb_is_grb (G_arg))
    G_arg = struct (G_arg) ;
end

% As an extension to the expression A(G), prune zeros and negative
% values first.  The expression A(G) becomes A (G (find (G > 0))).
gbmex_wait (G_arg) ;
G = gzb_select (1, '>0', G_arg) ;
gbmex_wait (G) ;

[m, n, type] = gbmex_size (G) ;
G_is_full = gb_isfull (G) ;

if (isequal (type, 'double') || isequal (type, 'single'))
    % double or single: convert to int64
    I = gbmex_extractvalues (G) ;
    if (~isequal (I, round (I)))
        error ('GrB:error', 'array indices must be integers') ;
    end
    I = int64 (I) ;
elseif (gb_contains (type, 'int'))
    % any integer: just extract the values
    I = gbmex_extractvalues (G) ;
else
    % logical or complex
    error ('GrB:error', 'array indices must be integers') ;
end

clear G

% I must contain entries in range 0 to prod (size (A)) - 1,
% so subtract the offset
I = I - 1 ;

% reshape I as needed
if (m == 1)
    % I should be a row vector instead
    I = I' ;
elseif (n > 1 && G_is_full)
    % I is should be an m-by-n matrix, so reshape it.  But I cannot be
    % reshaped to m-by-n if G is sparse, so leave it as a column vector.
    I = reshape (I, m, n) ;
end


