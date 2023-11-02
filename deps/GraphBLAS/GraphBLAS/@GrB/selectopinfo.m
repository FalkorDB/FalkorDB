function selectopinfo (op,optype)
%GRB.SELECTOPINFO list the details of a GraphBLAS select operator.
%
%   GrB.selectopinfo
%   GrB.selectopinfo (op)
%   GrB.selectopinfo (op, optype)
%
% For GrB.selectop(op), the op must be one of the following strings.
% Some of the operators have equivalent synonyms.
%
%   op                      built-in equivalent
%   --------                -----------------
%   'tril'                  C = tril (A,b)
%   'triu'                  C = triu (A,b)
%   'diag'                  C = diag (A,b)
%   'offdiag'               C = entries not in diag(A,b)
%   'rowne'                 C = A ; C (b,:) = 0
%   'rowle'                 C = A ; C (b+1:end,:) = 0
%   'rowgt'                 C = A ; C (1:b) = 0
%   'colne'                 C = A ; C (:,b) = 0
%   'colle'                 C = A ; C (:,b+1:end) = 0
%   'colgt'                 C = A ; C (:,1:b) = 0
%   '~=0' 'nonzero'         C = A (A ~= 0)
%   '==0' 'zero'            C = A (A == 0)
%   '>0'  'positive'        C = A (A >  0)
%   '>=0' 'nonnegative'     C = A (A >= 0)
%   '<0'  'negative'        C = A (A <  0)
%   '<=0' 'nonpositive'     C = A (A <= 0)
%   '~='                    C = A (A ~= b)
%   '=='                    C = A (A == b)
%   '>'                     C = A (A >  b)
%   '>='                    C = A (A >= b)
%   '<'                     C = A (A <  b)
%   '<='                    C = A (A <= b)
%
% Example:
%
%   GrB.selectopinfo ;
%   GrB.selectopinfo ('tril') ;
%
% See also GrB.binopinfo, GrB.descriptorinfo, GrB.monoidinfo,
% GrB.semiringinfo, GrB.unopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    help GrB.selectopinfo
elseif (nargin == 1)
    gbselectopinfo (op) ;
else
    gbselectopinfo (op, optype) ;
end

