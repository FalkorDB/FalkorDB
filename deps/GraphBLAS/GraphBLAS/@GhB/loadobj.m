function G = loadobj (S)
%LOADOBJ loads a GhB matrix from a file.
% The built-in MATLAB load method first reads in the struct S that saveobj
% created, and then passes it to this method.  Octave does not use this method
% since it cannot save/load objects to/from a file.
%
% See also GhB/saveobj, GhB.load.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% S cannot be a historical GhB object, since the GhB object was introduced in
% GraphBLAS v10.4.0, the same time loadobj and saveobj were added.  Thus, S
% must be a struct created by GhB/saveobj with a single S.blob field containing
% the serialized matrix.  

G = gzb_deserialize (1, S.blob) ;

