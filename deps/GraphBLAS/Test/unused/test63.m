function test63
%TEST63 test GraphBLAS binary operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
ops = binops.all ;
types = types.all ;

fprintf ('\n ---------------------------- testing GB_mex_op\n') ;

rng ('default') ;

n_operators = 0 ;
for k2 = 1:length(ops)
    mulop = ops {k2} ;
    if (GB_spec_is_positional (mulop))
        continue
    end
    fprintf ('\n%-10s ', mulop) ;

    for k1 = 1:length (types)
        type = types {k1} ;

        % create the op
        clear op
        op.opname = mulop ;
        op.optype = type ;

        try
            GB_spec_operator (op) ;
        catch
            continue ;
        end

        switch (mulop)
            case { 'pow' }
                xlimits = [0, 5] ;
                ylimits = [0, 5] ;
            case { 'ldexp' }
                xlimits = [-5, 5] ;
                ylimits = [-5, 5] ;
            otherwise
                xlimits = [ ] ;
                ylimits = [ ] ;
        end

        if (test_contains (type, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (type, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end

        fprintf ('.') ;

        for m = [ 1:3:10 ]% 100]
            for n = [1:3:10 ]% 100]
                for hi = [-1:5 200]
                    for lo = [-3:5 100]
                        A = full ((hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5));
                        B = full ((hi*sprand (m,n,0.8)-lo) .* sprand (m,n,0.5));

                        if (~isempty (xlimits))
                            A = max (A, xlimits (1)) ;
                            A = min (A, xlimits (2)) ;
                        end
                        if (~isempty (ylimits))
                            B = max (B, ylimits (1)) ;
                            B = min (B, ylimits (2)) ;
                        end

                        % use pure GraphBLAS
                        Z1 = GB_mex_op (op, A, B) ;
                        % use built-in methods as much as possible
                        Z2 = GB_spec_op (op, A, B) ;
                        % the results should either match perfectly
                        assert (isequal (isnan (Z1), isnan (Z2)))
                        assert (isequal (isfinite (Z1), isfinite (Z2)))
                        if (isequalwithequalnans (Z1, Z2))
                            continue ;
                        else
                            % ... or within roundoff error
                            mask = isfinite (Z1) ;
                            err = norm (Z1 (mask) - Z2 (mask), inf) ;
                            err = err / max (1, norm (Z1 (mask), inf)) ;
                            assert (err < tol) ;
                        end
                    end
                end
            end
        end
        n_operators = n_operators + 1  ;
    end
end

fprintf ('\nall tests passed\n') ;

fprintf ('Number of built-in GraphBLAS operators: %d\n',  n_operators) ;

fprintf ('\ntest63: all tests passed\n') ;

