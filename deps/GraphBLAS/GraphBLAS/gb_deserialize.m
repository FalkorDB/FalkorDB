function C = gb_deserialize (ghb, blob)
%GB_DESERIALIZE implements [GrB,GhB].deserialized.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (blob))
    blob = struct (blob) ;
end

% deserialize the blob into a GraphBLAS matrix
C = gzb_deserialize (ghb, blob) ;


