function L = gb_laplacian (ghb, A, type, check)
%GB_LAPLACIAN implements GrB.laplacian and GhB.laplacian.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

[m, n] = gbmex_size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square and symmetric') ;
end

% get the type
if (nargin < 3)
    type = 'double' ;
elseif (~gb_issigned (type))
    % type must be signed
    error ('GrB:error', 'type cannot be logical or unsigned integer') ;
end

% S = spones (A)
S = gzb_apply (1, ['1.' type], A) ;

% check the input matrix, if requested
if (nargin > 3 && isequal (check, 'check'))
    % make sure spones (S) is symmetric
    if (~gb_issymmetric (S, 'nonskew', false))
        error ('GrB:error', 'spones(A) must be symmetric') ;
    end
end

% D = diagonal matrix with d(i,i) = row/column degree of node i
if (gb_contains (gb_fmt (S), 'by row'))
    dim = 'row' ;
else
    dim = 'col' ;
end
D = gzb_mdiag (1, gzb_degree (1, S, 'dim'), 0) ;
if (~isequal (type, gb_type (D)))
    % gzb_degree returns its result as int64; typecast to desired type
    D = gzb (1, D, type) ;
end

% construct the Laplacian
% L = D-S
L = gzb_eadd (ghb, D, '+', gzb_apply (1, '-', S)) ;

