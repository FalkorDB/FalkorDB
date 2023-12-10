function monoidinfo (monoid, type)
%GRB.MONOIDINFO list the details of a GraphBLAS monoid.
%
%   GrB.monoidinfo
%   GrB.monoidinfo (monoid)
%   GrB.monoidinfo (monoid, type)
%
% For GrB.monoidinfo(op), the op must be a string of the form 'op.type',
% where 'op' is listed below.  The second usage allows the type to be
% omitted from the first argument, as just 'op'.  This is valid for all
% GraphBLAS operations, since the type defaults to the type of the input
% matrices.  However, GrB.monoidinfo does not have a default type and thus
% one must be provided, either in the op as GrB.monoidinfo ('+.double'), or
% in the second argument, GrB.monoidinfo ('+', 'double').
%
% A monoid is any binary operator z=f(x,y) that is commutative and
% associative, with an identity value o so that f(x,o)=f(o,x)=o.  The types
% of z, x, and y must all be identical.  For example, the plus.double
% operator is f(x,y)=x+y, with zero as the identity value (x+0 = 0+x = x).
% The times monoid has an identity value of 1 (since x*1 = 1*x = x).  The
% identity of min.double is +inf.
%
% The valid monoids for real non-logical types are:
%       '+', '*', 'max', 'min', 'any'
% For the 'logical' type:
%       '|', '&', 'xor', 'eq' (the same as 'xnor'), 'any'
% For complex types:
%       '+', '*', 'any'
% For integer types (signed and unsigned):
%       'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% Some monoids have synonyms; see 'help GrB.binopinfo' for details.
%
% Example:
%
%   % valid monoids
%   GrB.monoidinfo ('+.double') ;
%   GrB.monoidinfo ('*.int32') ;
%   GrB.monoidinfo ('min.double') ;
%
%   % invalid monoids
%   GrB.monoidinfo ('1st.int32') ;
%   GrB.monoidinfo ('abs.double') ;
%   GrB.monoidinfo ('min.complex') ;
%
% See also GrB.binopinfo, GrB.descriptorinfo, GrB.selectopinfo,
% GrB.semiringinfo, GrB.unopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    help GrB.monoidinfo
elseif (nargin == 1)
    gbmonoidinfo (monoid) ;
else
    gbmonoidinfo (monoid, type) ;
end

