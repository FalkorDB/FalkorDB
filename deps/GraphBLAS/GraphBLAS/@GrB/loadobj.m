function C = loadobj (S)
%LOADOBJ loads a GrB matrix from a file.
% The built-in MATLAB load method first reads in the struct S that saveobj
% created, and then passes it to this method.  Octave does not use this method
% since it cannot save/load objects to/from a file.
%
% See also GrB/saveobj, GrB.load.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (S))
    % S is a GrB matrix from GraphBLAS 10.3.1 or earlier, which
    % did not have saveobj and loadobj methods.
    if (gb_is_grb (S))
        S = struct (S) ;
    end
    C = gzb_loadhistorical (0, S) ;
else
    % S is a struct created by GrB/saveobj with a single
    % S.blob field containing the serialized matrix.
    C = gzb_deserialize (0, S.blob) ;
end

