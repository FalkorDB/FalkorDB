function test125
%TEST125 test GrB_mxm: row and column scaling
% all built-in semirings, no typecast, no mask

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
% mult_ops = binops.positional ;
mult_ops = binops.all ;
types = types.all ;

if (nargin < 1)
    fulltest = 1 ;
end

if (fulltest)
    fprintf ('-------------- GrB_mxm on all semirings (row,col scale)\n') ;
    n_semirings_max = inf ;
else
    fprintf ('quick test of GrB_mxm (dot product method)\n') ;
    n_semirings_max = 1 ;
end

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

ntrials = 0 ;

rng ('default') ;

n = 10 ;

n_semirings = 0 ;
A = GB_spec_random (n,n,0.3,100,'none') ;
clear B
B1matrix = spdiags (3 * rand (n,1), 0, n, n) ;
B.matrix = B1matrix ;
B.class = 'none' ;
B.pattern = logical (spones (B1matrix)) ;

C = GB_spec_random (n,n,0.3,100,'none') ;
M = spones (sprandn (n, n, 0.3)) ;

for k1 = 1:length(mult_ops)
    mulop = mult_ops {k1} ;
    if (fulltest)
        fprintf ('\n%-10s ', mulop) ;
    end
    nmult_semirings = 0 ;

    for k2 = 1:length(add_ops)
        addop = add_ops {k2} ;
        if (fulltest)
            fprintf ('.') ;
        end

        for k3 = 1:length (types)
            type = types {k3} ;

            semiring.multiply = mulop ;
            semiring.add = addop ;
            semiring.class = type ;

            % semiring

            % create the semiring.  some are not valid because the
            % or,and,xor monoids can only be used when z is boolean for
            % z=mult(x,y).
            try
                [mult_op add_op id] = GB_spec_semiring (semiring) ;
                [mult_opname mult_optype ztype xtype ytype] = ...
                    GB_spec_operator (mult_op) ;
                [ add_opname  add_optype] = GB_spec_operator (add_op) ;
                identity = GB_spec_identity (semiring.add, add_optype) ;
            catch
                continue
            end

            if (n_semirings+1 > n_semirings_max)
                fprintf ('\ntest125: all quick tests passed\n') ;
                return ;
            end

            n_semirings = n_semirings + 1 ;
            nmult_semirings = nmult_semirings + 1 ;
            A.class = type ;
            B.class = type ;
            C.class = type ;

            % C = A*B
            C1 = GB_mex_mxm  (C, [ ], [ ], semiring, A, B, dnn);
            C0 = GB_spec_mxm (C, [ ], [ ], semiring, A, B, dnn);
            GB_spec_compare (C0, C1, identity) ;

            % C = B*A
            C1 = GB_mex_mxm  (C, [ ], [ ], semiring, B, A, dnn);
            C0 = GB_spec_mxm (C, [ ], [ ], semiring, B, A, dnn);
            GB_spec_compare (C0, C1, identity) ;

            % dump the semiring list to compare with Source/FactoryKernels
            switch (xtype)
                case { 'logical' }
                    xtype = 'bool' ;
                case { 'single complex' }
                    xtype = 'fc32' ;
                case { 'double complex' }
                    xtype = 'fc64' ;
                case { 'single' }
                    xtype = 'fp32' ;
                case { 'double' }
                    xtype = 'fp64' ;
            end

            switch (add_opname)
                case { 'xor' }
                    add_opname = 'lxor' ;
                case { 'or' }
                    add_opname = 'lor' ;
                case { 'and' }
                    add_opname = 'land' ;
            end

            switch (mult_opname)
                case { 'xor' }
                    mult_opname = 'lxor' ;
                case { 'or' }
                    mult_opname = 'lor' ;
                case { 'and' }
                    mult_opname = 'land' ;
                case { 'pair', 'oneb' }
                    switch (add_opname)
                        case { 'eq', 'land', 'lor', 'min', 'max', 'times' }
                            add_opname = 'any' ;
                    end
            end
        end
    end
    fprintf (' %4d', nmult_semirings) ;
end

fprintf ('\nsemirings tested: %d\n', n_semirings) ;
fprintf ('\ntest125: all tests passed\n') ;

