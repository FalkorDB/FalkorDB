GraphBLAS/GraphBLAS/test/tcov:

The gbcov script compiles the interface with statement coverage
enabled, and then runs the full test suite (../gbtest).  Next, it uses
gbcovshow to create the statement coverage report in tmp/cover (one for
each file).  To remove all temporary files, use 'make distclean' or
remove the tmp/* files and folders manually.

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

