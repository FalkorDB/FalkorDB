function ok = isequal_roundoff (A,B,tol)
%ISEQUAL_ROUNDOFF compare two matrices, allowing for roundoff errors
% 
% returns true if A == B to within relative tolerance tol.
% tol = 64*eps if not present.  NaNs and Infs are ignored in the
% tol, but the NaN and +/-Inf pattern must be the same.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% if (~isequal (GB_spec_type (A), GB_spec_type (B)))
%     ok = false ;
%     return ;
% end

if (isequalwithequalnans (A, B))
    ok = true ;
    return
end

% floating-point NaN pattern must match exactly
if (~isequal (isnan (A), isnan (B)))
    % pattern of NaNs is different
    ok = false ;
    return
end

% remove NaNs
A (isnan (A)) = 0 ;
B (isnan (B)) = 0 ;

% floating-point Inf pattern must match exactly;
% treat +Inf and -Inf differently
if (~isequal (sign (A) .* isinf (A), sign (B) .* isinf (B)))
    % pattern of +Inf/-Inf is different
    ok = false ;
    return
end

% remove the Infs
A (isinf (A)) = 0 ;
B (isinf (B)) = 0 ;

% values must close, to within relative tol
err = norm (A - B, 1) ;
anorm = norm (A, 1) ;
if (nargin < 3)
    tol = 64*eps ;
end
anorm = max (anorm, 1) ;
ok = (err == 0) || (err <= tol * anorm) ;

