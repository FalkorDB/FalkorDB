function [arg1, arg2] = bandwidth (G, uplo)
%BANDWIDTH matrix bandwidth.
% [lo, hi] = bandwidth (G) returns the upper and lower bandwidth of G.
% lo = bandwidth (G, 'lower') returns just the lower bandwidth.
% hi = bandwidth (G, 'upper') returns just the upper bandwidth.
%
% See also GrB/isbanded, GrB/isdiag, GrB/istril, GrB/istriu.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (nargin == 1)
    % compute lo, and compute hi if present in output argument list
    [lo, hi] = gbmex_bandwidth (G, 1, nargout > 1) ;
    arg1 = lo ;
    arg2 = hi ;
else
    if (nargout > 1)
        error ('GrB:error', 'too many output arguments') ;
    elseif isequal (uplo, 'lower')
        [lo, ~] = gbmex_bandwidth (G, 1, 0) ;
        arg1 = lo ;
    elseif isequal (uplo, 'upper')
        [~, hi] = gbmex_bandwidth (G, 0, 1) ;
        arg1 = hi ;
    else
        error ('GrB:error', 'unrecognized option') ;
    end
end

