function C = gzb_split (ghb, A, m, n)
%GZB_SPLIT: wrapper for gbmex_split mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

% pre-allocate C to prevent memory failures after S is constructed
mlen = gb_length (m) ;
nlen = gb_length (n) ;
C = cell (mlen, nlen) ;
if (ghb)
    empty = zeros (1, 8, 'uint8') ;
    for k = 1:numel(C)
        C {k} = empty ;
    end
end

S = gbmex_split (ghb, A, m, n) ;

% convert each entry in S to a GrB or GhB object
if (ghb)
    for k = 1:numel(S)
        % NOTE: this method has a near zero chance of causing a memory leak
        % here.  If one of the conversions C {k} = GhB (S {k}) fails, this
        % method returns immediately.  MATLAB will know how to delete all GhB
        % objects in C by calling gbmex_delete.  It will not know how to
        % properly delete all of the GhB handle structs in S that remain.
        % These point to GraphBLAS matrices in malloc/free space, so this will
        % cause a leak.  However, this failure is very remote.  Each conversion
        % of GhB (S {k}) allocates a very small amount of memory and is
        % unlikely to fail.
        C {k} = GhB (S {k}) ;
    end
else
    for k = 1:numel(S)
        C {k} = GrB (S {k}) ;
    end
end

% sanity check to ensure C did not grow in size
assert (isequal (size (C), [mlen nlen])) ;

