function C = spfun (fun, G)
%SPFUN apply function to the entries of a matrix.
% C = spfun (fun, G) evaluates the function fun on the entries of G.
%
% If fun is a string, it can be any GraphBLAS unary operator (type
% 'help GrB.unopinfo' for a list).  The string can take the form of
% just the name of the operator ('sqrt', for example), or it can have
% the type appended, as ('sqrt.double', 'sqrt.double complex', etc).
% The latter produces a complex result.  sqrt.double (x) returns NaN if
% x is negative.  By default, the function type is determined from the
% type of the input matrix G.  GrB/spfun does not attempt to select the
% operator based on the values, in contract with GrB/sqrt, for example.
% For a list of types, see 'help GrB.type'.
%
% If the string fun is not a GraphBLAS operator, or if fun is a built-in
% function handle, then feval(fun,x) is used instead.
%
% The function is not applied to entries not present in G.  Since a
% GraphBLAS matrix can include explicit zeros, the function fun is
% applied to them as well.  Use GrB.prune to remove them, if necessary.
%
% Example:
%
%   A = sprand (4, 4, 0.5)
%   G = GrB (A) ;
%   Z = spfun ('exp', A)
%   Y = spfun ('exp', G)
%   C = exp (G)
%
%   % sqrt.double (-1) is nan:
%   z = spfun ('sqrt', GrB (-1))
%   % but sqrt.complex (-1) is 1i:
%   z = spfun ('sqrt', GrB (-1, 'complex'))
%   z = spfun ('sqrt.complex', GrB (-1))
%   % the overloaded GrB/sqrt function checks its inputs:
%   z = sqrt (GrB (2))
%   z = sqrt (GrB (-1))
%
% See also GrB.apply, GrB.unopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (G))
    G = G.opaque ;
end

if (ischar (fun))
    try
        C = GrB (gbapply (fun, G)) ;
        return ;
    catch me %#ok<NASGU>
        % gbapply failed; fall through to feval below
    end
end

% 'fun' is not a string, or not a built-in GraphBLAS operator
[m, n] = gbsize (G) ;
desc.base = 'zero-based' ;
[i, j, x] = gbextracttuples (G, desc) ;
x = feval (fun, x) ;
C = GrB.build (i, j, x, m, n, '1st', desc) ;

