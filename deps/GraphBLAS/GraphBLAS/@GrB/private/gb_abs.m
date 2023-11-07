function C = gb_abs (G)
%GB_ABS Absolute value of a GraphBLAS matrix.
% Implements C = abs (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_issigned (gbtype (G)))
    C = gbapply ('abs', G) ;
else
    C = G ;
end

