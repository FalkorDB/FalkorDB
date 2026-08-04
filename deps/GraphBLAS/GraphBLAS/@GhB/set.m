function set (G, state, value)
%GHB.SET set GhB matrix state.
% GhB.set (G, state, value) sets a state of a GhB matrix G.
%
%   state       value
%   'format'    See GrB.format.  The value is a string with any combination or
%               subset of 'sparse/hypersparse/bitmap' followed by 'by row' or
%               'by col' (if desired).  This changes how the matrix can be
%               stored.  For example, 'sparse/hypersparse by row' allows the
%               matrix to be stored in sparse or hypersparse format (GraphBLAS
%               selects between the two), held in a row-oriented data
%               structure.  The default is 'sparse/hyper/bitmap/full by col'
%               ('hyper' or 'hypersparse' mean the same thing).
%
%   'iso'       the value is a boolean, true or false.  If set to true, then
%               GraphBLAS will attempt to store the matrix as iso-valued, which
%               will succeed if all entries present have the same value.  The
%               data structure for an iso-valued matrix holds just one copy of
%               the value.  For example, GhB.ones(2^60) is a full matrix of
%               dimension 2^60 by 2^60, but it takes only a few 100 bytes to
%               store.  GraphBLAS typically detects the iso property itself,
%               but not all cases are caught.  Using GhB.set (G, 'iso', 'true')
%               will explicitly cause GraphBLAS to check.  If false, GraphBLAS
%               will change the matrix by expanding the iso- value scalar to
%               explicit individual entries.  This is sometimes preferable if
%               the matrix is about to become non-iso-valued anyway.
%
%   'offset', 'row', 'col' ('column'):  The value can be 32 or 64.  GraphBLAS
%               can use any mix of 32-bit or 64-bit integers for 3 kinds of
%               integer arrays: offsets, row indices, and column indices.  The
%               'offset' is an array in a sparse/hypersparse that references
%               the start of each row or column (see "doc mxGetJc" for the
%               equivalent for a MATLAB sparse matrix).  The row indices for a
%               matrix held by-column take up O(nvals(G)) space, which is cut
%               in half if 32-bit integers are used as compared to 64-bit
%               integers (see "doc mxGetIr" for the MATLAB equivalent).  Column
%               indices are used in a hypersparse matrix held by column (at
%               most n integers if G is m-by-n.  If G is held by row, there are
%               O(nvals(G)) column indices and up to O(m) row indices.  MATLAB
%               uses all-64-bit integers, and thus its sparse matrices use more
%               space than a GraphBLAS sparse matrix.
%
% This method has no outputs.  It changes the input matrix G in-place.  G must
% be a GhB GraphBLAS matrix.  It cannot be a GrB matrix or a built-in MATLAB/
% Octave matrix.
%
% See also GhB.format, GhB.bytes, GhB.isbyrow, GhB.isbycol.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

gbmex_set (G, state, value) ;

