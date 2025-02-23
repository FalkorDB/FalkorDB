function gbtest42
%GBTEST42 test for nan
%
% Also tests the JIT

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
types = { 'single', 'double', 'single complex', 'double complex' } ;

save_status = GrB.jit ;
states = { '', 'on', 'run', 'off', 'load', 'pause', 'flush', '', 'run' } ;

for nstate = 1:length(states)

    state = states {nstate} ;
    fprintf ('\nJIT: %s', state) ;
    new_state = GrB.jit (state) ;
    if (isequal (state, 'flush'))
        assert (isequal (new_state, 'on')) ;
    elseif (~isempty (state))
        assert (isequal (new_state, state)) ;
    end

    for k1 = 1:length(types)
        atype = types {k1} ;
        fprintf ('\n%s ', atype) ;

        for trial = 1:40
            fprintf ('.') ;

            A = gbtest_cast (full (sprand (4,4,0.5)), atype) ;
            A (A > 0.5) = nan ;

            A_nan = zeros (4, 4) ;
            A_nan (isnan (A)) = nan ;
            A_notnan = zeros (4, 4) ;
            A_notnan (~isnan (A)) = A (~isnan (A)) ;

            A = gbtest_cast (A, atype) ;

            for k2 = 1:length(types)
                xtype = types {k2} ;
                xnan = gbtest_cast (nan, xtype) ;

                G = GrB.select (A, '==', xnan) ;
                X1 = full (double (G)) ;
                X2 = double (A_nan) ;
                assert (isequaln (X1, X2)) ;

                G = GrB.select (A, '~=', xnan) ;
                X1 = full (double (G)) ;
                X2 = double (A_notnan) ;
                assert (isequaln (X1, X2)) ;

                G = GrB.prune (A, xnan) ;
                X1 = full (double (G)) ;
                X2 = double (A_notnan) ;
                assert (isequaln (X1, X2)) ;

            end
        end
    end
end

GrB.jit (save_status) ;

fprintf ('\ngbtest42: all tests passed\n') ;

