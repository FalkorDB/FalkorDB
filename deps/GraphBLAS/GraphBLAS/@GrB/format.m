function [f,s,iso] = format (arg)
%GRB.FORMAT get/set the default GraphBLAS matrix format.
%
% In its ANSI C interface, SuiteSparse:GraphBLAS stores its matrices by
% row, by default, since that format tends to be fastest for graph
% algorithms, but it can also store its matrices by column.  Built-in
% sparse and full matrices are always stored by column.  For better
% compatibility with built-in matrices, the default for this interface for
% SuiteSparse:GraphBLAS is to store matrices by column.  This has
% performance implications, and algorithms should be designed accordingly.
% The default format can be can changed via:
%
%   GrB.format ('by row')
%   GrB.format ('by col')
%
% which changes the format of all subsequent GraphBLAS matrices.
% Existing GrB matrices are not affected.
%
% The current default global format can be queried with
%
%   f = GrB.format ;
%
% which returns the string 'by row' or 'by col'.
%
% Converting a matrix to a specific format can be done with the following,
% where A is either a GraphBLAS matrix or built-in matrix:
%
%   G = GrB (A, 'by row')
%
% If a subsequent algorithm works better with its matrices held by row,
% then this transformation can save significant time in the long run.
% Graph algorithms tend to be faster with their matrices held by row,
% since the edge (i,j) is typically the entry G(i,j) in the matrix G, and
% most graph algorithms need to know the outgoing edges of node i.  This
% is G(i,:), which is very fast if G is held by row, but very slow if G
% is held by column.
%
% When the GrB.format (f) is changed, it becomes the default format for
% all subsequent newly created matrices.
%
% All prior matrices created before GrB.format (f) are kept in their same
% format; this setting only applies to new matrices.  Operations on
% matrices can be done with any mix of with different formats.  The format
% only affects time and memory usage, not the results.
%
% The format of the output C of a GraphBLAS method is defined using the
% following rules.  The first rule that holds is used:
%
%   (1) GraphBLAS operations of the form C = GrB.method (Cin, ...)
%       that take a Cin input matrix, use the format of Cin as the format
%       for C, if Cin is provided on input.
%   (2) If the format is determined by the descriptor to the method, then
%       that determines the format of C.
%   (3) If C is a column vector then C is stored by column.
%   (4) If C is a row vector then C is stored by row.
%   (5) If the method has a first matrix input (usually called A), and it
%       is not a row or column vector, then its format is used for C.
%   (6) If the method has a second matrix input (usually called B), and
%       it is not a row or column vector, then its format is used for C.
%   (7) Otherwise, the global default format is used for C.
%
% The GrB.format setting is reset to its default ('by col'), via GrB.clear.
%
% To query the format for a given GraphBLAS matrix G, use the following
% (which does not affect the global format setting).  The return value f
% is 'by row' or 'by col', s is 'hypersparse', 'sparse', 'bitmap',
% or 'full', and iso is 'iso-valued' or 'non-iso-valued'.
%
%   [f,s,iso] = GrB.format (G)
%
% Use G = GrB (G, 'by row') or G = GrB (G, 'by col') to change the format
% of G after it is constructed.
%
% Individual matrices are held in one of four data structurs, each of
% which can be held 'by row' and 'by col'.  By default, GraphBLAS selects
% automatically between the following four formats.  Let A by m-by-n with
% e entries:
%
%   (1) 'hypersparse' (or 'hyper' for short):  This is useful if A
%       n << e and A is 'by col', or m << e if A is 'by row'.  The data
%       structure takes only O (e) space.
%   (2) 'sparse':  This the same as the built-in sparse matrix, except that
%       A can be either 'by col' (taking O(n+e) space), or 'by row'
%       taking O(m+e) space.  A native built-in sparse matrix is only held
%       'by col'.
%   (3) 'bitmap':  This data structure takes O(m*n) space, but it can
%       represent a sparse matrix with e < m*n.  It is very efficient
%       if e is about 0.1*m*n or greater.
%   (4) 'full':  This takes O(m*n) sparse, and 'full by col' is the same
%       as a built-in full matrix.  All entries must be present (e == m*n).
%       GraphBLAS can also store a matrix 'full by row'.
%
% The sparsity formats can be combined.  For example, to store a matrix in
% either sparse or bitmap format (but not hypersparse or full) use G = GrB
% (A, 'sparse/bitmap by col').  GraphBLAS will automatically select
% between the 'sparse by col' and 'bitmap by col' formats, choosing the
% latter if the density e/m*n exceeds a default threshold b.  A bitmap
% matrix is converted to sparse if its density drops below the b/2.  The
% value of b depends on min(m,n).  A matrix between these two ranges is
% kept in its current format.  The with 'sparse/bitmap by col' format, a
% matrix will not be held in hypersparse or full formats.  The default is
% 'hyper/sparse/bitmap/full by col', which allows GraphBLAS to select
% between all 4 formats, each column-oriented 'by col'.
%
% Examples:
%
%   A = sparse (rand (4))
%   G = GrB (A)                  % format always 'by col'
%   G = GrB (A, 'by row')        % change to 'by row'
%   GrB.format (G)
%   GrB.format ('by row') ;      % set the default format to 'by row'
%   G = GrB.build (1:3, 1:3, 1:3)
%   [f,s,iso] = GrB.format (G)   % query the format of G
%   C = GrB.ones (2^60)          % now that's a big matrix
%   [f,s,iso] = GrB.format (C)
%
% See also GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    % f = GrB.format ; get the global format
    if (nargout > 1)
        error ('GrB:error', 'usage: f = GrB.format') ;
    end
    f = gbformat ;
else
    if (isobject (arg))
        % f = GrB.format (G) ; get the format of the GraphBLAS matrix
        arg = arg.opaque ;
    end
    % f = GrB.format (A) ; get the format of A (built-in or GraphBLAS)
    % f = GrB.format (f) ; set the global format for all matrices.
    if (nargout <= 1)
        f = gbformat (arg) ;
    else
        [f,s,iso] = gbformat (arg) ;
    end
end

