function blob = serialize (G, method, level)
%GRB.SERIALIZE convert a matrix to a serialized blob.
% blob = GrB.serialize (G) returns a uint8 array containing the contents
% of the matrix G, which may be a MATLAB or @GrB matrix.  The array may
% be saved to a binary file and used to construct a GrB_Matrix outside
% of this MATLAB/Octave interface to GraphBLAS.  It may also be used to
% reconstruct a @GrB matrix with G = GrB.deserialize (blob).
%
% blob = GrB.serialize (G,method,level) specifies the compression method,
% as a string.  The 3rd parameter is optional; it is an integer that
% specifices the compression level, with a higher level resulting in a
% more compact blob at the cost of higher run time.  Levels outside
% the allowable range are changed to the default level.
%
%   'zstd'  ZSTD.  The level can be 1 to 19 with 1 the default.
%           This is the default method if no method is specified.
%
%   'lz4'   LZ4, with no level setting. Fast with decent compression.
%           For large problems, lz4 can be faster than no compression,
%           and it cuts the size of the blob by about 3x on average.
%
%   'none'  no compression.
%
%   'lz4hc' LZ4HC, much slower than LZ4 but results in a more compact blob.
%           The level can be 1 to 9 with 9 the default.  LZ4HC level 1
%           provides excellent compression compared with LZ4, and higher
%           levels of LZ4HC only slightly improve compression quality.
%
% Example:
%   G = GrB (magic (5))
%   blob = GrB.serialize (G) ;      % compressed via ZSTD, level 1
%   f = fopen ('G.bin', 'wb') ;
%   fwrite (f, blob) ;
%   fclose (f)
%   clear all
%   f = fopen ('G.bin', 'r') ;
%   blob = fread (f, '*uint8') ;
%   G = GrB.deserialize (blob)
%
% See also GrB.deserialize, GrB.load, GrB.save, GrB/struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (G))
    % extract the contents of a GraphBLAS matrix
    G = G.opaque ;
end

% serialize the matrix into a uint8 blob
if (nargin == 1)
    % use the default compression method and default level
    blob = gbserialize (G) ;
elseif (nargin == 2)
    % use the given compression method and default level
    blob = gbserialize (G, method) ;
else
    % use the given compression method and given level
    blob = gbserialize (G, method, level) ;
end

