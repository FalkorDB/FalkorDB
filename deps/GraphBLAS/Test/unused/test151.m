function test151
%TEST151 test bitwise operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test151: test bitwise operators\n') ;

[binops, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.int ;
ops2 = binops.int ;

int_nbits = [ 8, 16, 32, 64, 8, 16, 32, 64 ] ;

rng ('default') ;
Cin = sparse (4,4) ;
C10 = sparse (10,10) ;

for k = 1:8

    type = types {k} ;
    nbits = int_nbits (k) ;
    fprintf ('\n%s', type) ;

    for trial = 1:40
        fprintf ('.') ;

        % dense case

        imax = double (intmax (type) / 4) ;
        A = GB_mex_cast (imax * rand (4), type) ;
        B = GB_mex_cast ((nbits-1) * rand (4), type) + 1 ;
        clear A2 ; A2.matrix = sparse (double (A)) ; A2.class = type ;
        clear B2 ; B2.matrix = sparse (double (B)) ; B2.class = type ;
        A2.pattern = logical (spones (A)) ;
        B2.pattern = logical (spones (B)) ;

        for j = 1:length (ops2)
            opname = ops2 {j} ;
            % C1 = bitop (A, B) ;
            op.opname = opname ; op.optype = type ;

            if (isequal (opname, 'bitshift') || isequal (opname, 'bshift'))
                B2.class = 'int8' ;
            else
                B2.class = type ;
            end

            C1 = GB_spec_Matrix_eWiseMult(Cin, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;
        end

        % C1 = bitcmp (A) ;
        op.opname = 'bitnot' ; op.optype = type ;
        C1 = GB_spec_apply(Cin, [ ], [ ], op, A2, [ ]) ;
        C2 = GB_mex_apply (Cin, [ ], [ ], op, A2, [ ]) ;
        GB_spec_compare (C1, C2) ;

        % sparse case

        A = sprand (10, 10, 0.5) * imax ;
        Afull = GB_mex_cast (full (A), type) ;
        % B ranges in value from 0 to 8
        B = round (sprand (10, 10, 0.5) * nbits) ;
        Bfull = GB_mex_cast (full (B), type) ;
        clear A2 ; A2.matrix = sparse (double (Afull)) ; A2.class = type ;
        clear B2 ; B2.matrix = sparse (double (Bfull)) ; B2.class = type ;
        A2.pattern = logical (spones (Afull)) ;
        B2.pattern = logical (spones (Bfull)) ;

        for j = 1:length (ops2)
            opname = ops2 {j} ;
            % C1 = bitop (A, B) ;
            op.opname = opname ; op.optype = type ;

            if (isequal (opname, 'bitshift') || isequal (opname, 'bshift'))
                B2.class = 'int8' ;
            else
                B2.class = type ;
            end

            C1 = GB_spec_Matrix_eWiseMult(C10, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseMult (C10, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (C10, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (C10, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion (C10, [ ], [ ], op, A2, 3, B2, 1, [ ]) ;
            C2 = GB_mex_Matrix_eWiseUnion  (C10, [ ], [ ], op, A2, 3, B2, 1, [ ]) ;
            GB_spec_compare (C1, C2) ;

        end

        % C1 = bitcmp (Afull) ;
        op.opname = 'bitnot' ; op.optype = type ;
        C1 = GB_spec_apply(C10, [ ], [ ], op, A2, [ ]) ;
        C2 = GB_mex_apply (C10, [ ], [ ], op, A2, [ ]) ;
        GB_spec_compare (C1, C2) ;

    end
end

fprintf ('\ntest151: all tests passed\n') ;

