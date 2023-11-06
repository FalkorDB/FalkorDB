function semiringinfo (s, type)
%GRB.SEMIRINGINFO list the details of a GraphBLAS semiring.
%
%   GrB.semiringinfo
%   GrB.semiringinfo (semiring)
%   GrB.semiringinfo (semiring, type)
%
% For GrB.semiring(semiring), the semiring must be a string of the form
% 'add.mult.type', where 'add' and 'mult' are binary operators.  The
% second usage allows the type to be omitted from the first argument, as
% just 'add.mult'.  This is valid for all GraphBLAS operations, since the
% type defaults to the type of the input matrices.  However,
% GrB.semiringinfo does not have a default type and thus one must be
% provided, either in the semiring as GrB.semiringinfo ('+.*.double'), or
% in the second argument, GrB.semiringinfo ('+.*', 'double').
%
% The additive operator must be the binary operator of a valid monoid (see
% 'help GrB.monoidinfo').  The multiplicative operator can be any binary
% operator z=f(x,y) listed by 'help GrB.binopinfo', but the type of z must
% match the operand type of the monoid.  The type in the string
% 'add.mult.type' is the type of x for the multiply operator z=f(x,y), and
% the type of its z output defines the type of the monoid.
%
% Example:
%
%   % valid semirings
%   GrB.semiringinfo ('+.*.double') ;
%   GrB.semiringinfo ('min.1st.int32') ;
%
%   % invalid semiring (generates an error; since '<' is not a monoid)
%   GrB.semiringinfo ('<.*.double') ;
%
% See also GrB.binopinfo, GrB.descriptorinfo, GrB.monoidinfo,
% GrB.selectopinfo, GrB.unopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    help GrB.semiringinfo
elseif (nargin == 1)
    gbsemiringinfo (s) ;
else
    gbsemiringinfo (s, type) ;
end

