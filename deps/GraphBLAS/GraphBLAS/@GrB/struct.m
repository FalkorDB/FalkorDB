function S = struct (G)
%GRB.STRUCT return the opaque (private) contents of a @GrB object.
% S = struct (G) returns the opaque properties of a @GrB object as a
% struct.  The properties of G are private, and thus the fields of S
% cannot be interpretted except by GrB (S).  However, S can be can be
% saved to a file and then loaded back in, which may facilitate
% portability with future SuiteSparse:GraphBLAS versions, in case the
% contents of a @GrB object changes.  S can be converted back into a @GrB
% object with G = GrB (S).
%
% See also load, save, properties, GrB.version, GrB.ver, GrB.load, GrB.save.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

S = G.opaque ;
