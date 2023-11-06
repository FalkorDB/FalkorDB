function test248
%TEST248 count the number of valid built-in semirings
% This does not count all possible semirings that could be constructed from
% all built-in types and operators.  Just those that are in GraphBLAS.h,
% or equivalent to those in that list.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
mult_ops = binops.all ;
types = types.all ;

n_semirings = 0 ;

for k1 = 1:length(mult_ops)
    mulop = mult_ops {k1} ;
%   fprintf ('\n%-10s ', mulop) ;
    nmult_semirings = 0 ;

    for k2 = 1:length(add_ops)
        addop = add_ops {k2} ;
%       fprintf ('.') ;

        for k3 = 1:length (types)
            type = types {k3} ;

%           fprintf ('\n---------- try this: %s.%s.%s\n', addop, mulop, type) ;
            semiring.add = addop ;
            semiring.multiply = mulop ;
            semiring.class = type ;

            % create the semiring.  some are not valid because the
            % or,and,xor monoids can only be used when z is boolean for
            % z=mult(x,y).

            try
                [mult_op add_op id] = GB_spec_semiring (semiring) ;
                [mult_opname mult_optype ztype xtype ytype] = ...
                    GB_spec_operator (mult_op) ;
                [ add_opname  add_optype] = GB_spec_operator (add_op) ;
                identity = GB_spec_identity (semiring.add, add_optype) ;
            catch me
                continue
            end

            n_semirings = n_semirings + 1 ;
            nmult_semirings = nmult_semirings + 1 ;

            % This produces a superset of the list of all files in
            % Source/FactoryKernels, if the operators and are renamed ("xor"
            % becomes "lxor", "logical" becomes "bool", etc).  Some semirings
            % have different names but are handled by boolean renaming.  Other
            % semirings appear here, and in GraphBLAS.h, but do not have any
            % kernel in FactoryKernels (GxB_TIMES_LOR_FP64 for example).
%           semiring
%           fprintf ('GB_AxB__%s_%s_%s.c\n', add_opname, mult_opname, xtype) ;
        end
    end

%   fprintf (' %d', nmult_semirings) ;

end

fprintf ('\nTotal: %d semirings\n', n_semirings) ;

