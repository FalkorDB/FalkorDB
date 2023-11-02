function test102
%TEST102 test GB_AxB_saxpy3_flopcount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest102: testing GB_AxB_saxpy3_flopcount\n') ;

rng ('default') ;

for m = [0 1 10 100]
    for n = [0 1 10 100]
        for d = [0.01 0.1 0.5 1.0]

            % create the mask M and its hypersparse version
            M = sprand (m, n, d) ;
            Mhyper.matrix = M ;
            Mhyper.pattern = spones (M) ;
            Mhyper.is_hyper = true ;

            for k = [0 1 10 100]

                % create B and its hypersparse version
                B = sprand (k, n, d) ;
                Bhyper.matrix = B ;
                Bhyper.pattern = spones (B) ;
                Bhyper.is_hyper = true ;

                % create A and its hypersparse version
                A = sprand (m, k, d) ;
                Ahyper.matrix = A ;
                Ahyper.pattern = spones (A) ;
                Ahyper.is_hyper = true ;

                % flop counts for C=A*B
                mflops = flopcount ([ ], 0, A, B) ;
                total = mflops (end) ;

                floptest ([ ], 0, A, B, mflops) ;
                floptest ([ ], 0, Ahyper, Bhyper, total) ;
                floptest ([ ], 0, Ahyper, B, total) ;
                floptest ([ ], 0, A, Bhyper, total) ;

                % flop counts for C<M>=A*B
                mflops = flopcount (M, 0, A, B) ;
                total = mflops (end) ;

                floptest (M, 0, A, B, mflops) ;
                floptest (M, 0, A, Bhyper, total) ;
                floptest (M, 0, Ahyper, B, total) ;
                floptest (M, 0, Ahyper, Bhyper, total) ;
                floptest (Mhyper, 0, A, B, total) ;
                floptest (Mhyper, 0, Ahyper, B, total) ;
                floptest (Mhyper, 0, A, Bhyper, total) ;
                floptest (Mhyper, 0, A, B, total) ;

            end
        end
    end
end

fprintf ('\ntest102: all tests passed\n') ;

