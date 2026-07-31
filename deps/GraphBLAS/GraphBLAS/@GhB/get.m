function value = get (G, state)
%GHB.GET get matrix state.
% value = GhB.get (G, state) gets the current state of a matrix.  The state
% input is a string: 'format', 'iso', 'offset', 'row', 'col' (or 'column').
%
% value = GhB.get (G, 'format') returns the current allowed sparsity formats
% the matrix G may be held in.  It is not just the current format, which is
% what [f,s,iso] = GhB.format (G) returns.  The default format is
% 'sparse/hypersparse/bitmap/full by col', which means G can take on any of the
% 4 formats, all in a column-oriented storage.
%
% value = GhB.get (G, 'iso') returns true if the matrix is currently held in 
% an iso-valued format, false otherwise.
%
% value = GhB.get (G, 'offset') returns 32 or 64; see GhB.set.
% value = GhB.get (G, 'row') returns 32 or 64; see GhB.set.
% value = GhB.get (G, 'col') returns 32 or 64; see GhB.set.
%
% Example:
%
%   G = GhB.ones (4)            % creates a 4-by-4 iso-valued matrix
%   s = GhB.get (G, 'iso')      % returns s as true
%   G (1,1) = 3                 % G is no longer iso-valued
%   s = GhB.get (G, 'iso')      % returns s as false
%   G (1,1) = 1
%   s = GhB.get (G, 'iso')      % returns s as still false since this is one
%                               % of the cases that GraphBLAS does not detect
%                               % the iso property of a matrix; it would be too
%                               % costly in general.
%   GhB.set (G, 'iso', 1)       % G is now detected as iso-valued, and GraphBLAS
%                               % changes the data structure of G, reducing
%                               % memory usage.
%
% The input to GhB.get G can be any GhB, GrB, or built-in MATLAB/Octave matrix.
% Only GhB matrices can be used with GhB.set however.
%
% See also GhB.format, GhB.bytes, GhB.isbyrow, GhB.isbycol.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

value = gbmex_get (G, state) ;

