function [I, J, X] = find (G, k, search)
%FIND extract entries from a matrix.
% [I, J, X] = find (G) extracts the nonzeros from a matrix G.
% X has the same type as G ('double', 'single', 'int8', ...).
%
% Linear 1D indexing (I = find (S) for the built-in matrix S) is not yet
% supported.
%
% A GraphBLAS matrix G may contain explicit zero entries, and by default
% these are excluded from the result.  Use GrB.extracttuples (G) to return
% these explicit zero entries.
%
% For a column vector, I = find (G) returns I as a list of the row indices
% of nonzeros in G.  For a row vector, I = find (G) returns I as a list of
% the column indices of nonzeros in G.
%
% [...] = find (G, k, 'first') returns the first k nonozeros of G.
% [...] = find (G, k, 'last')  returns the last k nonozeros of G.
% For this usage, the first and last k are in terms of nonzeros in the
% column-major order.
%
% See also sparse, GrB.build, GrB.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: find (G,k,'first') and find (G,k,'last') are slow.
% They are currently implemented, all entries are extracted and then the
% first or last k are selected from the extracted tuples.  It would be
% faster to use a mexFunction that directly accesses the opaque content
% of G, instead of using GrB_Matrix_extractTuples_*, which always extracts
% the entire matrix.

if (isobject (G))
    G = G.opaque ;
end

if (nargin > 1)
    k = ceil (double (gb_get_scalar (k))) ;
    if (k < 1)
        error ('GrB:error', 'k must be positive') ;
    end
    if (~isequal (gbformat (G), 'by col'))
        % find (G, k) assumes the matrix is stored by column, so reformat G
        % if it is stored by row.
        G = gbnew (G, 'by col') ;
    end
end

% prune explicit zeros
G = gbselect (G, 'nonzero') ;

[m, n] = gbsize (G) ;

if (nargout == 3)
    [I, J, X] = gbextracttuples (G) ;
    if (m == 1)
        I = I' ;
        J = J' ;
        X = X' ;
    end
elseif (nargout == 2)
    [I, J] = gbextracttuples (G) ;
    if (m == 1)
        I = I' ;
        J = J' ;
    end
else
    if (m == 1)
        % extract indices from a row vector
        [~, I] = gbextracttuples (G) ;
        I = I' ;
    elseif (n == 1)
        % extract indices from a column vector
        I = gbextracttuples (G) ;
    else
        % extract linear indices from a matrix
        [I, J] = gbextracttuples (G) ;
        % use the built-in sub2ind to convert the 2D indices to linear indices
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

