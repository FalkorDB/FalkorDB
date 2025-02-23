function result = selectops
%GRB.SELECTOPS list all select ops
% Redundant select ops are not listed.  For example 'tril.double'
% exists, but it is identical to just 'tril'.
%
% Example:
%   GrB.selectops ;         % prints a list, with descriptions
%   list = GrB.selectops ;  % returns the list (nothing printed)
%
% See also GrB.selectopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = {
    'logical'
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    'single complex'
    'double complex'
    } ;

selectops = {
'nonzero'       , 'keeps nonzero entries, removes zeros' ;
'zero'          , 'keeps zeros, removes nonzeros' ;
'positive'      , 'keeps entries > 0, removes entries <= 0' ;
'nonnegative'   , 'keeps entries >= 0, removes entries < 0' ;
'negative'      , 'keeps entries < 0, removes entries >= 0' ;
'nonpositive'   , 'keeps entries <= 0, removes entries > 0' ;
'tril'          , 'keeps entries in tril(A,b)' ;
'triu'          , 'keeps entries in triu(A,b)' ;
'diag'          , 'keeps entries in diag(A,b)' ;
'offdiag'       , 'keeps entries not in diag(A,b)' ;
'rowne'         , 'keeps entries not in A(b,:), removes entries in A(b,:)' ;
'rowle'         , 'keeps entries in A(1:b,:), removes entries in A(b+1:end,:)' ;
'rowgt'         , 'keeps entries in A(b+1:end,:), removes entries in A(1:b,:)' ;
'colne'         , 'keeps entries not in A(:,b), removes entries in A(:,b)' ;
'colle'         , 'keeps entries in A(:,1:b), removes entries in A(:,b+1:end)' ;
'colgt'         , 'keeps entries in A(:,b+1:end), removes entries in A(:,1:b)' ;
'~='            , 'keeps entries not equal to b, removes entries equal to b' ;
'=='            , 'keeps entries equal to b, removes entries not equal to b' ;
'>'             , 'keeps entries entries > b, removes entries <= b' ;
'>='            , 'keeps entries entries >= b, removes entries < b' ;
'<'             , 'keeps entries entries < b, removes entries >= b' ;
'<='            , 'keeps entries entries <= b, removes entries > b' } ;

nselectops = 0 ;
nops = size (selectops, 1) ;

if (nargout > 0)
    result = { } ;
end

    for k2 = 1:nops
        op = selectops {k2,1} ;
        op_description = selectops {k2,2} ;
        first_op= true ;

        ignore_type = gb_contains (op, 'tri') || gb_contains (op, 'diag') || ...
            gb_contains (op, 'col') || gb_contains (op, 'row') ;
        if (ignore_type)
            ntypes = 1 ;
        else
            ntypes = length (types) ;
        end

        for k3 = 1:ntypes
            ok = false ;
            if (ignore_type)
                selectop = op ;
            else
                type = types {k3} ;
                selectop = [op '.' type] ;
            end

            try
                ok = gbselectopinfo (selectop) ;
                nselectops = nselectops + 1 ;
                if (nargout > 0)
                    result = [result ; selectop] ; %#ok<AGROW>
                end
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct
                % a valid selectop.
            end
            if (ok && nargout == 0)
                if (ignore_type)
                    fprintf ('select op: %s', op) ;
                    fprintf (', %s', op_description) ;
                elseif (first_op)
                    fprintf ('select op: %s.type', op) ;
                    fprintf (', %s\n', op_description) ;
                    fprintf ('        types: %s', type) ;
                else
                    fprintf (', %s', type) ;
                end
                first_op = false ;
            end
        end
        if (nargout == 0)
            fprintf ('\n\n') ;
        end
    end

if (nargout == 0)
    fprintf ('Total number of available select ops: %d\n', nselectops) ;
end

