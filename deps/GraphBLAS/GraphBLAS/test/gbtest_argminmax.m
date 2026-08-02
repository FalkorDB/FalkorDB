function [x,p] = gbtest_argminmax (ghb, A, ismin, dim)
%GBTEST_ARGMINMAX simple computation of argmin and argmax.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

S = spones (A) ;
[m n] = size (A) ;
type = gtb_type (ghb, A) ;
p = [ ] ;

if (dim == 2)

    [x,p] = gbtest_argminmax (ghb, A', ismin, 1) ;

elseif (dim == 1)

    x = gtb (ghb, n, 1, type) ;
    p = gtb (ghb, n, 1, 'int64') ;
    for j = 1:n
        first = true ;
        for i = 1:m
            % octave requires the explicit cast to logical for an "if(...)"
            if (logical (S (i,j) == 1))
                if (first)
                    x (j) = A (i,j) ;
                    p (j) = i ;
                    first = false ;
                else
                    if (ismin)
                        if (logical (A (i,j) < x (j)))
                            x (j) = A (i,j) ;
                            p (j) = i ;
                        end
                    else
                        if (logical (A (i,j) > x (j)))
                            x (j) = A (i,j) ;
                            p (j) = i ;
                        end
                    end
                end
            end
        end
    end

else % dim == 0

    x = gtb (ghb, n, 1, type) ;
    p = gtb (ghb, n, 1, 'int64') ;
    first = true ;
    for i = 1:m
        for j = 1:m
            if (logical (S (i,j) == 1))
                if (first)
                    x = A (i,j) ;
                    p = [i j] ;
                    first = false ;
                else
                    if (ismin)
                        if (logical (A (i,j) < x))
                            x = A (i,j) ;
                            p = [i j] ;
                        end
                    else
                        if (logical (A (i,j) > x))
                            x = A (i,j) ;
                            p = [i j] ;
                        end
                    end
                end
            end
        end
    end
    p = gtb (ghb, p', 'int64') ;

end

