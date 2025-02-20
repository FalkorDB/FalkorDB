function test188
%TEST188 test concat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test188 ----------- C = concat (Tiles)\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

track_coverage = true ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
    cstart = find (GraphBLAS_grbcov > 0) ;
end

rng ('default') ;

n1 = 20 ;
n2 = 4 ;

densities = [1e-4 0.01 0.2 0.8 inf] ;
for k1 = 1:length(densities)
    density = densities (k1) ;
    % fprintf ('\nd = %g\n', density) ;

    for ka = 1:length (types)
        atype = types {ka} ;

        A1 = GB_spec_random (n1, n1, density, 128, atype) ;
        A2 = GB_spec_random (n1, n2, density, 128, atype) ;
        A3 = GB_spec_random (n2, n1, density, 128, atype) ;

        if (ka == 11)
            A4 = GB_spec_random (n2, n2, inf, 128, atype) ;
        else
            A4 = GB_spec_random (n2, n2, density, 128, atype) ;
        end

        for iso = 0:1

            % test iso case
            A1.iso = iso ;
            A2.iso = iso ;
            A3.iso = iso ;
            A4.iso = iso ;

            for sparsity_control = [1 2 4 8]
                A1.sparsity = sparsity_control ;
                A2.sparsity = sparsity_control ;
                A3.sparsity = sparsity_control ;
                if (ka == 11)
                    A4.sparsity = 8 ;
                else
                    A4.sparsity = sparsity_control ;
                end
                for is_csc = [0 1]
                    A1.is_csc = is_csc ;
                    A2.is_csc = is_csc ;
                    A3.is_csc = is_csc ;
                    A4.is_csc = is_csc ;
                    Tiles = cell (2,2) ;
                    Tiles {1,1} = A1 ;
                    Tiles {1,2} = A2 ;
                    Tiles {2,1} = A3 ;
                    Tiles {2,2} = A4 ;
                    for kc = 1:length (types)
                        ctype = types {kc} ;
                        for fmt = 0:1
                            C1 = GB_mex_concat  (Tiles, ctype, fmt) ;
                            C2 = GB_spec_concat (Tiles, ctype) ;
                            GB_spec_compare (C1, C2) ;

                            if (track_coverage)
                                c = sum (GraphBLAS_grbcov > 0) ;
                                d = c - clast ;
                                if (d > 0)
                                    fprintf (...
                                    '[%15s, %15s, %d, %d, %d, %d, %d]', ...
                                    ctype, atype, ...
                                    k1, iso, sparsity_control, ...
                                    is_csc, fmt) ;
                                    fprintf (' (%d, %d)\n', d, c - cfirst) ;
                                end
                                clast = c ;
                            else
                                fprintf ('.') ;
                            end

                        end
                    end
                end
            end
        end
    end
end

fprintf ('\n') ;
GB_mex_burble (0) ;
fprintf ('test188: all tests passed\n') ;

