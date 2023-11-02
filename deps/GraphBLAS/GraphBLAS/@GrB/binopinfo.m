function binopinfo (op, optype)
%GRB.BINOPINFO list the details of a GraphBLAS binary operator.
%
%   GrB.binopinfo
%   GrB.binopinfo (op)
%   GrB.binopinfo (op, optype)
%
% Binary operators are defined by a string of the form 'op.optype', or
% just 'op', where the optype is inferred from the operands.  Valid
% optypes are 'logical', 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32', 'uint64', 'single', 'double', 'single complex',
% 'double complex' (the latter can be written as simply 'complex').
%
% For GrB.binopinfo (op), the op must be a string of the form 'op.optype',
% where 'op' is listed below.  The second usage allows the optype to be
% omitted from the first argument, as just 'op'.  This is valid for all
% GraphBLAS operations, since the optype can be determined from the
% operands (see Typecasting, below).  However, GrB.binopinfo does not have
% any operands and thus the optype must be provided, either in the op as
% GrB.binopinfo ('+.double'), or in the second argument as
% GrB.binopinfo ('+', 'double').
%
% The 6 comparator operators come in two flavors.  For the is* operators,
% the result has the same type as the inputs, x and y, with 1 for true and
% 0 for false.  For example isgt.double (pi, 3.0) is the double value 1.0.
% For the second set of 6 operators (eq, ne, gt, lt, ge, le), the result
% is always logical (true or false).  In a semiring, the optype of the add
% monoid must exactly match the type of the output of the multiply
% operator, and thus 'plus.iseq.double' is valid (counting how many terms
% are equal).  The 'plus.eq.double' semiring is valid, but not the same
% semiring since the 'plus' of 'plus.eq.double' has a logical type and is
% thus equivalent to 'or.eq.double'.   The 'or.eq' is true if any terms
% are equal and false otherwise (it does not count the number of terms
% that are equal).
%
% The following binary operators are available for most types.  Many have
% equivalent synonyms, so that '1st' and 'first' both define the
% first(x,y) = x operator.
%
%   operator name(s) f(x,y)         |   operator names(s) f(x,y)
%   ---------------- ------         |   ----------------- ------
%   1st first        x              |   iseq              x == y
%   2nd second       y              |   isne              x ~= y
%   min              min(x,y)       |   isgt              x > y
%   max              max(x,y)       |   islt              x < y
%   +   plus         x+y            |   isge              x >= y
%   -   minus        x-y            |   isle              x <= y
%   rminus           y-x            |   ==  eq            x == y
%   *   times        x*y            |   ~=  ne            x ~= y
%   /   div          x/y            |   >   gt            x > y
%   \   rdiv         y/x            |   <   lt            x < y
%   |   || or  lor   x | y          |   >=  ge            x >= y
%   &   && and land  x & y          |   <=  le            x <= y
%   xor lxor         xor(x,y)       |   .^  pow           x .^ y
%   pair oneb        1              |   any               pick x or y
%
% All of the above operators are defined for logical operands, but many
% are redundant. 'min.logical' is the same as 'and.logical', for example.
% Most of the logical operators have aliases: ('lor', 'or', '|') are the
% same, as are ('lxnor', 'xnor', 'eq', '==') for logical types.
%
% Positional operators return int32 or int64, and depend only on the position
% of the entry in the matrix.  They do not depend on the values of their
% inputs, but on their position in the matrix instead:
%
%   1-based postional ops:          in a semiring:     in ewise operators:
%   operator name(s)                f(A(i,k)*B(k,j))   f(A(i,j),B(i,j))
%   ----------------                ----------------   ----------------
%   firsti1  1sti1 firsti  1sti     i                  i
%   firstj1  1stj1 firstj  1stj     k                  j
%   secondi1 2ndi1 secondi 2ndi     k                  i
%   secondj1 2ndj1 secondj 2ndj     j                  j
%
%   0-based postional ops:          in a semiring:     in ewise operators:
%   operator name(s)                f(A(i,k)*B(k,j))   f(A(i,j),B(i,j))
%   ----------------                ----------------   ----------------
%   firsti0  1sti0                  i-1                i-1
%   firstj0  1stj0                  k-1                j-1
%   secondi0 2ndi0                  k-1                i-1
%   secondj0 2ndj0                  j-1                j-1
%
% Comparators (*lt, *gt, *le, *ge) and min/max are not available for
% complex types.
%
% The three logical operators, lor, land, and lxor, can be used with any
% real types.  z = lor.double (x,y) tests the condition (x~=0) || (y~=0),
% and returns the double value 1.0 if true, or 0.0 if false.
%
% The following operators are avaiable for single and double (real); their
% definitions are identical to the C11 versions of these functions:
% atan2, hypot, fmod, remainder, copysign, ldxep (also called 'pow2').
% All produce the same type as the input, on output.
%
% z = cmplx(x,y) can be computed for x and y as single and double; z is
% single complex or double complex, respectively.
%
% The bitwise ops bitor, bitand, bitxor, bitxnor, bitget, bitset, bitclr,
% and bitshift are available for any signed or unsigned integer type.
%
% The following index_unary operators can be applied to a matrix A with
% GrB.apply2 only, where the 2nd input is the thunk scalar.  When applied
% to an entry A(i,j):
%
%   tril            j <= (i + thunk)
%   triu            j >= (i + thunk)
%   diag            j == (i + thunk)
%   offdiag         j != (i + thunk)
%   diagindex       j - (i + thunk)
%   rowindex        i + thunk
%   rowle           i <= thunk
%   rowgt           i > thunk
%   colindex        j + thunk
%   colle           j <= thunk
%   colgt           j > thunk
%
% Typecasting:  If the optype is omitted from the string (for example,
% GrB.eadd (A, '+', B) or simply C = A+B), then the optype is inferred
% from the type of A and B.  See 'help GrB.optype' for details.
%
% Example:
%
%   % valid binary operators
%   GrB.binopinfo ('+.double') ;    % also a valid unary operator
%   GrB.binopinfo ('1st.int32') ;
%   GrB.binopinfo ('cmplx.single') ;
%   GrB.binopinfo ('pow2.double') ; % also a valid unary operator
%   GrB.unopinfo  ('pow2.double') ;
%
%   % invalid binary operator (an error; this is a unary op):
%   GrB.binopinfo ('abs.double') ;
%
% See also GrB.descriptorinfo, GrB.monoidinfo, GrB.selectopinfo,
% GrB.semiringinfo, GrB.unopinfo, GrB.optype.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    help GrB.binopinfo
elseif (nargin == 1)
    gbbinopinfo (op) ;
else
    gbbinopinfo (op, optype) ;
end

