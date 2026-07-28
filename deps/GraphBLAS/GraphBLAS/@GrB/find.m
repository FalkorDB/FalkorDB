function [I, J, X] = find (G_arg, k, search)
%FIND extract entries from a matrix.
% [I, J, X] = find (G) extracts the nonzeros from a matrix G.  I, J, and X are
% built-in matrices, not GraphBLAS matrices.  X has the same type as G.
%
% Linear 1D indexing (I = find (S) for the built-in matrix S) is not yet
% supported.
%
% A GraphBLAS matrix G may contain explicit zero entries, and by default these
% are excluded from the result.  Use GrB.extracttuples (G) to return these
% explicit zero entries.
%
% For a column vector, I = find (G) returns I as a list of the row indices of
% nonzeros in G.  For a row vector, I = find (G) returns I as a list of the
% column indices of nonzeros in G.
%
% [...] = find (G, k, 'first') returns the first k nonozeros of G.
% [...] = find (G, k, 'last')  returns the last k nonozeros of G.  For this
% usage, the first and last k are in terms of nonzeros in the column-major
% order.  Note that this usage is much slower than when using a MATLAB/Octave
% built-in matrix, because a GraphBLAS matrix must have the ability to hold
% explicit zeros.  These must be pruned to match the behavior of find(G,k),
% which takes extra work.
%
% The indices I and J are returned as int32 or int64 column vectors, depending
% on the dimenions of the matrix G.
%
% See also sparse, GrB.build, GrB.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: add an option to return I,J,X as GrB or GhB matrices.

if (gb_is_grb (G_arg))
    G_arg = struct (G_arg) ;
end

if (nargin > 1)
    k = ceil (double (gb_get_scalar (k))) ;
    if (k < 1)
        error ('GrB:error', 'k must be positive') ;
    end
    % prune explicit zeros and ensure G is held by col
    desc.format = 'by col' ;
    G = gzb_select (1, G_arg, 'nonzero', desc) ;
else
    % prune explicit zeros and leave G in the same format
    G = gzb_select (1, G_arg, 'nonzero') ;
end

[m, n] = gbmex_size (G) ;
gbmex_wait (G) ;

if (nargout == 3)
    [I, J, X] = gbmex_extracttuples (1, G) ;
    if (m == 1)
        I = I' ;
        J = J' ;
        X = X' ;
    end
elseif (nargout == 2)
    [I, J] = gbmex_extracttuples (1, G) ;
    if (m == 1)
        I = I' ;
        J = J' ;
    end
else
    if (m == 1)
        % extract indices from a row vector
        [~, I] = gbmex_extracttuples (1, G) ;
        I = I' ;
    elseif (n == 1)
        % extract indices from a column vector
        I = gbmex_extracttuples (1, G) ;
    else
        % extract linear indices from a matrix
        [I, J] = gbmex_extracttuples (1, G) ;
        % use the built-in sub2ind to convert the 2D indices to 1D indices
        I = sub2ind ([m n], I, J) ;
    end
end

if (nargin > 1)
    % find (G, k, ...): get the first or last k entries
    if (nargin < 3)
        search = 'first' ;
    end
    n = length (I) ;
    if (k >= n)
        % output already has all k first or last entries;
        % nothing more to do
    elseif (isequal (search, 'first'))
        % find (G, k, 'first'): get the first k entries
        I = I (1:k) ;
        if (nargout > 1)
            J = J (1:k) ;
        end
        if (nargout > 2)
            X = X (1:k) ;
        end
    elseif (isequal (search, 'last'))
        % find (G, k, 'last'): get the last k entries
        I = I (n-k+1:n) ;
        if (nargout > 1)
            J = J (n-k+1:n) ;
        end
        if (nargout > 2)
            X = X (n-k+1:n) ;
        end
    else
        error ('GrB:error', ...
            'invalid search option; must be ''first'' or ''last''') ;
    end
end

