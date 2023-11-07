function C = angle (G)
%ANGLE phase angle.
% C = angle (G) is the phase angle of each entry of G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
[m, n, type] = gbsize (G) ;

if (gb_contains (type, 'complex'))
    C = GrB (gbapply ('carg', G)) ;
else
    % C is all zero
    C = GrB (gbnew (m, n, type)) ;
end

