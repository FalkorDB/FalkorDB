function C = deserialize (blob)
%GRB.DESERIALIZE convert a serialized blob into a matrix.
% C = GrB.deserialize (blob) returns a GrB matrix constructed from the uint8
% array blob constructed by GhB.serialize or GrB.serialize.
%
% C = GrB.deserialize (blob) or GrB.deserialize (blob, 'fast') assume the blob
% comes from a trusted source.  C = GrB.deserialize (blob, 'secure') does a
% secure (but slow) deserialization, checking the blob to ensure it is valid,
% when the blob might not be trusted.
%
% Example:
%
%   C = GrB (magic (5))
%   blob = GrB.serialize (C) ;
%   f = fopen ('C.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('C.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   C = GrB.deserialize (blob)
%
% See also GrB.serialize, GrB.load, GrB.save, GrB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_deserialize (0, blob) ;

