function G = deserialize (blob, mode, arg3)
%GRB.DESERIALIZE convert a serialized blob into a matrix.
% G = GrB.deserialize (blob) returns a @GrB matrix constructed from the
% uint8 array blob constructed by GrB.serialize.
%
% G = GrB.deserialize (blob) or GrB.deserialize (blob, 'fast') assume the
% blob comes from a trusted source.  G = GrB.deserialize (blob, 'secure')
% does a secure (but slow) deserialization, checking the blob to ensure it
% is valid, when the blob might not be trusted.
%
% Example:
%   G = GrB (magic (5))
%   blob = GrB.serialize (G) ;
%   f = fopen ('G.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('G.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   G = GrB.deserialize (blob)
%
% See also GrB.serialize, GrB.load, GrB.save, GrB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% The type may also be specified, as an optional arg3 string:
% G = GrB.deserialize (blob, 'fast', 'single') for example.  This is
% feature not documented, and is only intended for testing.  The ANSI C
% rules for casting floating-point to integers are used (truncation), not
% the MATLAB rules (rounding to nearest integer).

% deserialize the blob into a @GrB matrix
if (nargin == 1)
    G = GrB (gbdeserialize (blob)) ;
elseif (nargin == 2)
    G = GrB (gbdeserialize (blob, mode)) ;
else
    G = GrB (gbdeserialize (blob, mode, arg3)) ;
end

