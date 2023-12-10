function C = ceil (G)
%CEIL round entries of a matrix to nearest integers towards infinity.
%
% See also GrB/floor, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Q = G.opaque ;

if (gb_isfloat (gbtype (Q)) && gbnvals (Q) > 0)
    C = GrB (gbapply ('ceil', Q)) ;
else
    C = G ;
end

