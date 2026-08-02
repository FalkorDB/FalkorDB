function C = deserialize (blob)
%GHB.DESERIALIZE convert a serialized blob into a matrix.
% C = GhB.deserialize (blob) returns a GhB matrix constructed from the uint8
% array blob constructed by GhB.serialize or GrB.serialize.
%
% C = GhB.deserialize (blob) or GhB.deserialize (blob, 'fast') assume the blob
% comes from a trusted source.  C = GhB.deserialize (blob, 'secure') does a
% secure (but slow) deserialization, checking the blob to ensure it is valid,
% when the blob might not be trusted.
%
% Example:
%
%   C = GhB (magic (5))
%   blob = GhB.serialize (C) ;
%   f = fopen ('C.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('C.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   C = GhB.deserialize (blob)
%
% See also GhB.serialize, GhB.load, GhB.save, GhB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_deserialize (1, blob) ;

