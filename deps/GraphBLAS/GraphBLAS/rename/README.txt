This folder contains a Makefile that extracts the external symbols in
libgraphblas.so, and creates an #include file that renames them to a different
namespace.  It is required to install to use GraphBLAS with MATLAB R2021a and
later.

To create the GB_rename.h file, first compile libgraphblas.so
Then type "make" in this folder.  This should only be required to develop
GraphBLAS since the GB_rename.h file is included in the source distribution.

Linux is required, but the resulting GB_rename.h is then used by all platforms.

