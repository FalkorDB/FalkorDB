function [I,J,X] = extracttuples (A, desc)
%GRB.EXTRACTTUPLES extract a list of entries from a matrix.
%
%   [I,J,X] = GrB.extracttuples (A, desc)
%
% GrB.extracttuples or GhB.extracttuples extract all entries from either a
% built-in or GraphBLAS matrix.  If A is a built-in full or sparse matrix,
% [I,J,X] = GrB.extracttuples (A) is identical to [I,J,X] = find (A).
%
% For a GraphBLAS matrix G, GrB.extracttuples (G) returns any explicit zero
% entries in G, while find (G) excludes them.
%
% The descriptor is optional.  desc.base is a string, either 'default',
% 'zero-based', 'one-based int', 'one-based', 'double' or 'one-based double'.
% This determines the type of output for I and J.  The default is 'one-based
% int', so that I and J are returned as int32 or int64 vectors, with one-based
% indices.  For 'double', or 'one-based double', then I and J are returned as
% double, unless the dimensions are > flintmax, in which case they are returned
% as int64.  One-based indices in I are in the range 1 to m, and the indices in
% J are in the range 1 to n, if A is m-by-n.  This is identical to [I,J,X] =
% find (A) for a built-in sparse or full MATLAB matrix, except that the MATLAB
% find returns I and J as double.
%
% If 'zero-based', I and J are returned as int32 or int64 arrays, with
% zero-based indices.  The entries in I and J are in the range 0 to m-1 and 0
% to n-1, respectively, if [m n] = size (A).  This usage is not the
% conventional 1-based indexing, but it is the fastest method.
%
% The overloaded [I,J,X] = find (A) method for a GraphBLAS matrix A uses
% desc.base of 'default', and always removes explicit zeros.
%
% I, J, and X are returned as built-in matrices, not GrB or GhB matrices.
%
% See also GrB/find, GrB/build.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 2)
    desc.base = 'default' ;
end

% gbmex_extracttuples requires A to have no pending work
gbmex_wait (A) ;

switch (nargout)
    case 1
        I = gbmex_extracttuples (1, A, desc) ;
    case 2
        [I, J] = gbmex_extracttuples (1, A, desc) ;
    case 3
        [I, J, X] = gbmex_extracttuples (1, A, desc) ;
end

