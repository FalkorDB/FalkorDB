function s = issigned (arg)
%GRB.ISSIGNED Determine if a type is signed or unsigned.
% s = GrB.issigned (type) returns true if type is the string 'double',
% 'single', 'single complex', 'double complex', 'int8', 'int16', 'int32',
% or 'int64'.
%
% s = GrB.issigned (A), where A is a matrix, is the same as
% s = GrB.issigned (GrB.type (A)).
%
% See also GrB/isinteger, GrB/isreal, GrB/isnumeric, GrB/isfloat, GrB.type.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ischar (arg))
    type = arg ;
elseif (isobject (arg))
    arg = arg.opaque ;
    type = gbtype (arg) ;
else
    type = gbtype (arg) ;
end

s = gb_issigned (type) ;

