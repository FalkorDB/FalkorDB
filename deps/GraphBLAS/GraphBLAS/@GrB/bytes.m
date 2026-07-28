function mem = bytes (A)
%GRB.BYTES the # of bytes used by a matrix.
%
% mem = bytes (A) returns the # of bytes used by a GrB, GhB, or builtin
% matrix.  The memory statistics can vary a few 100 bytes from the 'whos'
% report because of how GraphBLAS constructs its matrices as MATLAB objects.
%
% See also GrB/size, GrB/nvals, GrB.isbycol, GrB.isbyrow, GhB.set, GhB.get.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end
[~, ~, ~, mem] = gbmex_size (A) ;

