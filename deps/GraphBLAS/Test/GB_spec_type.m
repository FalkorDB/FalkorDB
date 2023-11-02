function s = GB_spec_type (X)
%GB_SPEC_TYPE determine the class of a built-in matrix
% s = GB_spec_type (X) determines the class of X.  It is identical to s =
% class(X), except when X is a complex matrix.  In this case the string
% 'complex' is appended to the result of class (X).  If X is a single and
% iscomplex (X) is true, then s is 'single complex', and if X double, s is
% 'double complex'.  Complex integer matrices result in 'int8 complex' (for
% example), although GraphBLAS does not currently handle those.
%
% For a built-in matrix X, GB_spec_type (X) and GrB.type (X) are identical.
%
% See also GrB.type.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = class (X) ;
if (~isreal (X))
    s = [s ' complex'] ;
end

