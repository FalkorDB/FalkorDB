% benchmark @GrB vs built-in methods

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

dolegend

dobench_methods ('ND/nd3k', 4) ;
dobench_methods ('ND/nd24k', 4) ;
dobench_methods ('SNAP/roadNet-CA', 3) ;
% dobench_methods ('SNAP/soc-Pokec', 3) ;
dobench_methods ('Freescale/Freescale2', 3) ;
dobench_methods ('LAW/indochina-2004', 3) ;
dobench_methods ('SNAP/com-Orkut', 3) ;
dobench_methods ('GAP/GAP-road', 3) ;

dobench_methods ('GAP/GAP-twitter', 3) ;
dobench_methods ('GAP/GAP-web', 2) ;
dobench_methods ('GAP/GAP-urand', 1) ;
dobench_methods ('GAP/GAP-kron', 1) ;

