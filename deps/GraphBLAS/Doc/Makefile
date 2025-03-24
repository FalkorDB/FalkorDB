#-------------------------------------------------------------------------------
# GraphBLAS/Doc/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# creates the user guide

GraphBLAS_UserGuide.pdf: GraphBLAS_UserGuide.tex GraphBLAS_UserGuide.bib \
    GraphBLAS_API_version.tex GraphBLAS_version.tex Makefile \
    UserGuide/GrB_abstract.tex \
    UserGuide/GrB_acknowledgements.tex \
    UserGuide/GrB_colon.tex \
    UserGuide/GrB_concepts.tex \
    UserGuide/GrB_context.tex \
    UserGuide/GrB_examples.tex \
    UserGuide/GrB_get_set_BinaryOp.tex \
    UserGuide/GrB_get_set_Context.tex \
    UserGuide/GrB_get_set_Descriptor.tex \
    UserGuide/GrB_get_set_enum.tex \
    UserGuide/GrB_get_set_Global.tex \
    UserGuide/GrB_get_set_IndexBinaryOp.tex \
    UserGuide/GrB_get_set_IndexUnaryOp.tex \
    UserGuide/GrB_get_set_integers.tex \
    UserGuide/GrB_get_set_Matrix.tex \
    UserGuide/GrB_get_set_Monoid.tex \
    UserGuide/GrB_get_set_Scalar.tex \
    UserGuide/GrB_get_set_Semiring.tex \
    UserGuide/GrB_get_set_Serialize.tex \
    UserGuide/GrB_get_set.tex \
    UserGuide/GrB_get_set_Type.tex \
    UserGuide/GrB_get_set_UnaryOp.tex \
    UserGuide/GrB_get_set_Vector.tex \
    UserGuide/GrB_init.tex \
    UserGuide/GrB_installing.tex \
    UserGuide/GrB_interfaces.tex \
    UserGuide/GrB_intro.tex \
    UserGuide/GrB_iso.tex \
    UserGuide/GrB_iterator.tex \
    UserGuide/GrB_jit.tex \
    UserGuide/GrB_mask.tex \
    UserGuide/GrB_matlab_performance.tex \
    UserGuide/GrB_objects_BinaryOp.tex \
    UserGuide/GrB_objects_Container.tex \
    UserGuide/GrB_objects_Descriptor.tex \
    UserGuide/GrB_objects_formats.tex \
    UserGuide/GrB_objects_free.tex \
    UserGuide/GrB_objects_import.tex \
    UserGuide/GrB_objects_IndexBinaryOp.tex \
    UserGuide/GrB_objects_IndexUnaryOp.tex \
    UserGuide/GrB_objects_Matrix.tex \
    UserGuide/GrB_objects_Monoid.tex \
    UserGuide/GrB_objects_Scalar.tex \
    UserGuide/GrB_objects_Semiring.tex \
    UserGuide/GrB_objects_Serialize.tex \
    UserGuide/GrB_objects_sort.tex \
    UserGuide/GrB_objects.tex \
    UserGuide/GrB_objects_Type.tex \
    UserGuide/GrB_objects_UnaryOp.tex \
    UserGuide/GrB_objects_Vector.tex \
    UserGuide/GrB_operations_apply.tex \
    UserGuide/GrB_operations_assign_duplicates.tex \
    UserGuide/GrB_operations_assign.tex \
    UserGuide/GrB_operations_assign_vs_subassign.tex \
    UserGuide/GrB_operations_eWiseAdd.tex \
    UserGuide/GrB_operations_eWiseMult.tex \
    UserGuide/GrB_operations_eWiseUnion.tex \
    UserGuide/GrB_operations_extract.tex \
    UserGuide/GrB_operations_kronecker.tex \
    UserGuide/GrB_operations_mxm.tex \
    UserGuide/GrB_operations_mxv.tex \
    UserGuide/GrB_operations_reduce.tex \
    UserGuide/GrB_operations_select.tex \
    UserGuide/GrB_operations_subassign.tex \
    UserGuide/GrB_operations.tex \
    UserGuide/GrB_operations_transpose.tex \
    UserGuide/GrB_operations_vxm.tex \
    UserGuide/GrB_performance.tex \
    UserGuide/GrB_print.tex \
    UserGuide/GrB_references.tex \
    UserGuide/GrB_release.tex \
    UserGuide/GrB_resources.tex \
    UserGuide/GrB_v10.tex
	pdflatex GraphBLAS_UserGuide.tex
	bibtex GraphBLAS_UserGuide
	pdflatex GraphBLAS_UserGuide.tex
	pdflatex GraphBLAS_UserGuide.tex
	pdflatex GraphBLAS_UserGuide.tex

clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc

purge: clean

distclean: clean

