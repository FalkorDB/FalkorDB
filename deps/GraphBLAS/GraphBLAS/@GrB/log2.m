function [F, E] = log2 (G)
%LOG2 base-2 logarithm.
% C = log2 (G) is the base-2 logarithm of each entry of a GraphBLAS matrix
% G.  Since log2 (0) is nonzero, the result is a full matrix.  If any entry
% in G is negative, the result is complex.
%
% [F,E] = log2 (G) returns F and E so that G = F.*(2.^E), where entries in
% abs (F) are either in the range [0.5,1), or zero if the entry in G is
% zero.  F and E are both sparse, with the same pattern as G.  If G is
% complex, [F,E] = log2 (real (G)).
%
% See also GrB/pow2, GrB/log, GrB/log1p, GrB/log10, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (nargout == 1)
    % C = log2 (G)
    F = GrB (gb_check_imag_zero (gb_trig ('log2', gbfull (G)))) ;
else
    % [F,E] = log2 (G)
    type = gbtype (G) ;
    switch (type)
        case { 'logical', 'int8', 'int16', 'int32', 'int64', ...
            'uint8', 'uint16', 'uint32', 'uint64', 'double complex' }
            type = 'double' ;
        case { 'single complex' }
            type = 'single' ;
        case { 'single', 'double' }
            % type remains the same
    end
    F = GrB (gbapply (['frexpx.' type], G)) ;
    E = GrB (gbapply (['frexpe.' type], G)) ;
end

