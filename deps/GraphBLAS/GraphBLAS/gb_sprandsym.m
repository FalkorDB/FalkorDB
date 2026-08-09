function C = gb_sprandsym (ghb, varargin)
%GB_SPRANDSYM implements GrB/sprandsym and GhB/sprandsym.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (2, 3) ;

for k = 1:numel (varargin)
    if (gb_is_grb (varargin {k}))
        varargin {k} = struct (varargin {k}) ;
    end
end

if (nargin == 2)
    % C = sprandsym (G)
    C = gb_random (ghb, varargin {1}, 'symmetric', 'normal') ;
else
    % C = sprandsym (n, d)
    n = gb_get_scalar (varargin {1}) ;
    d = gb_get_scalar (varargin {2}) ;
    C = gb_random (ghb, n, d, 'symmetric', 'normal') ;
end


