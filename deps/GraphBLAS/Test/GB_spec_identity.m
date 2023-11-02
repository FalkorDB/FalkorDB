function identity = GB_spec_identity (arg1,arg2)
%GB_SPEC_IDENTITY the additive identity of a monoid
%
% identity = GB_spec_identity (add) ;
% or
% identity = GB_spec_identity (add_op, add_type) ; % both strings
%
% Returns the additive identity of an operator of a monoid.
%
% The identity is that value such that x == add (x,identity) == add
% (identity,x).  For example, for 'plus', the value is zero.  for 'max' the
% value of identity is -infinity.
%
% The addititive monoids supported are 'min', 'max', 'plus', 'times', 'or',
% 'and', 'xor', 'eq', 'bitor', 'bitand', 'bitxor', and 'bitxnor'.
% For the 'or', 'and', 'xor', and 'eq' the add_type must be 'logical'.
% For the 'bit*' operators, the add_type must be unsigned integer.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    add = arg1 ;
    [add_operator add_type ztype xtype ytype] = GB_spec_operator (add) ;
else
    add_operator = arg1 ;
    add_type = arg2 ;
end

switch add_operator

    case 'min'

        % x == min (x, inf)
        switch add_type
            case 'logical'
                identity = true ;
            case 'int8'
                identity = int8 (inf) ;
            case 'uint8'
                identity = uint8 (inf) ;
            case 'int16'
                identity = int16 (inf) ;
            case 'uint16'
                identity = uint16 (inf) ;
            case 'int32'
                identity = int32 (inf) ;
            case 'uint32'
                identity = uint32 (inf) ;
            case 'int64'
                identity = int64 (inf) ;
            case 'uint64'
                identity = uint64 (inf) ;
            case 'single'
                identity = single (inf) ;
            case 'double'
                identity = inf ;
        end

    case 'max'

        % x == max (x, -inf)
        switch add_type
            case 'logical'
                identity = false ;
            case 'int8'
                identity = int8 (-inf) ;
            case 'uint8'
                identity = 0 ;
            case 'int16'
                identity = int16 (-inf) ;
            case 'uint16'
                identity = 0 ;
            case 'int32'
                identity = int32 (-inf) ;
            case 'uint32'
                identity = 0 ;
            case 'int64'
                identity = int64 (-inf) ;
            case 'uint64'
                identity = 0 ;
            case 'single'
                identity = single (-inf) ;
            case 'double'
                identity = -inf ;
        end

    case 'plus'

        % x == x + 0
        identity = 0 ;

    case 'times'

        % x == x * 1
        identity = 1 ;

    case 'any'

        identity = [ ] ;

    case { 'or', 'lor'}

        % x == x or false
        identity = false ;

        if (~isequal (add_type, 'logical'))
            error ('OR monoid must be logical') ;
        end

    case { 'and', 'land' }

        % x == x and true
        identity = true ;

        if (~isequal (add_type, 'logical'))
            error ('AND monoid must be logical') ;
        end

    case { 'xor', 'lxor' }

        % x == x xor false
        identity = false ;

        if (~isequal (add_type, 'logical'))
            error ('XOR monoid must be logical') ;
        end

    case { 'eq', 'lxnor'}

        % x == (x == true)
        identity = true ;

        if (~isequal (add_type, 'logical'))
            error ('EQ monoid must be logical') ;
        end

    case { 'iseq', 'eq' }

        % x == (x == true)
        identity = true ;

        if (~isequal (add_type, 'logical'))
            error ('ISEQ monoid must be logical') ;
        end

    case { 'bitor', 'bor' }

        switch add_type
            case { 'uint8', 'uint16', 'uint32', 'uint64' }
                identity = 0 ;
            otherwise
                error ('BIT* monoids must be unsigned int') ;
        end

    case { 'bitand', 'band' }

        switch add_type
            case { 'uint8', 'uint16', 'uint32', 'uint64' }
                identity = intmax (add_type) ;
            otherwise
                error ('BIT* monoids must be unsigned int') ;
        end

    case { 'bitxor', 'bxor' }

        switch add_type
            case { 'uint8', 'uint16', 'uint32', 'uint64' }
                identity = 0 ;
            otherwise
                error ('BIT* monoids must be unsigned int') ;
        end

    case { 'bitxnor', 'bxnor' }

        switch add_type
            case { 'uint8', 'uint16', 'uint32', 'uint64' }
                identity = intmax (add_type) ;
            otherwise
                error ('BIT* monoids must be unsigned int') ;
        end

    otherwise

        error ('unsupported additive monoid') ;
end

identity = GB_mex_cast (identity, add_type) ;

