function p = amd (G, opts)
%AMD approximate minimum degree ordering.
% See 'help amd' for details.
%
% See also GrB/colamd, GrB/symrcm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    p = amd (logical (G)) ;
else
    p = amd (logical (G), opts) ;
end

