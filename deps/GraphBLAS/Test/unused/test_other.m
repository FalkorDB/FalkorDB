function test_other
%TEST_OTHER installs all packages needed for extensive tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

here = pwd ;
fprintf ('\n------------------installing ssget\n') ;
try
    index = ssget ;
catch
    cd ../../ssget
    addpath (pwd) ;
end
cd (here) ;

fprintf ('\n------------------installing spok\n') ;
cd spok
addpath (pwd) ;
try
    spok (sparse (1)) ;
catch
    spok_install ;
end
cd (here) ;

fprintf ('\n------------------installing SSMULT\n') ;
cd ../../SuiteSparse/MATLAB_Tools/SSMULT
addpath (pwd) ;
try
    L = sparse (1) ;
    ssmultsym (L, L) ;
catch
    ssmult_install
end
cd (here) ;

fprintf ('\n------------------installing CXSparse\n') ;
cd ../../SuiteSparse/CXSparse/MATLAB/CSparse
addpath (pwd) ;
try
    cs_sparse (1, 1, 1) ;
catch
    cs_make (1) ;
end
cd (here) ;

fprintf ('\n------------------installing CHOLMOD\n') ;
cd ../../SuiteSparse/CHOLMOD/MATLAB
addpath (pwd) ;
try
    sparse2 (1, 1, 1) ;
catch
    cholmod_make ;
end
cd (here) ;

