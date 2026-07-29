function C = gb_sprand (ghb, dist, varargin)
%GB_SPRAND implementes GrB/sprand and GhB/sprand.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (3, 5) ;

for k = 1:numel (varargin)
    if (gb_is_grb (varargin {k}))
        varargin {k} = struct (varargin {k}) ;
    end
end

if (nargin == 3)
    % C = sprand (G) or sprandn (G)
    C = gb_random (ghb, varargin {1}, dist) ;
elseif (nargin == 5)
    % C = sprand (m, n, d) or sprandn (m, n, d)
    m = gb_get_scalar (varargin {1}) ;
    n = gb_get_scalar (varargin {2}) ;
    d = gb_get_scalar (varargin {3}) ;
    C = gb_random (ghb, m, n, d, dist) ;
else
    % the 'rc' input option is not supported
    error ('GrB:error', 'usage: rc input option is not supported') ;
end

