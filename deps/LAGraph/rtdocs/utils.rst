Utility Functions
=================

Input/Output Functions
----------------------

.. _lagraph_mmread:
.. doxygenfunction:: LAGraph_MMRead

.. _lagraph_mmwrite:
.. doxygenfunction:: LAGraph_MMWrite

.. _lagraph_wallclocktime:
.. doxygenfunction:: LAGraph_WallClockTime

Matrix Structure Functions
--------------------------

.. doxygenfunction:: LAGraph_Matrix_Structure

.. doxygenfunction:: LAGraph_Vector_Structure

Matrix Comparison Functions
---------------------------

.. doxygenfunction:: LAGraph_Matrix_IsEqual

.. doxygenfunction:: LAGraph_Matrix_IsEqualOp

.. doxygenfunction:: LAGraph_Vector_IsEqual

.. doxygenfunction:: LAGraph_Vector_IsEqualOp


Introspecting Types
-------------------

.. doxygenfunction:: LAGraph_NameOfType

.. doxygenfunction:: LAGraph_TypeFromName

.. doxygenfunction:: LAGraph_SizeOfType

.. doxygenfunction:: LAGraph_Matrix_TypeName

.. doxygenfunction:: LAGraph_Vector_TypeName

.. doxygenfunction:: LAGraph_Scalar_TypeName


Printing
--------

.. doxygenfunction:: LAGraph_Graph_Print

.. doxygenfunction:: LAGraph_Matrix_Print

.. doxygenfunction:: LAGraph_Vector_Print

.. doxygenenum:: LAGraph_PrintLevel

Pre-defined semirings
---------------------

LAGraph adds the following pre-defined semirings.  They are created
by :ref:`LAGr_Init <lagr_init>` or :ref:`LAGraph_Init <lagraph_init>`,
and freed by :ref:`LAGraph_Finalize <lagraph_finalize>`.

- LAGraph_plus_first_T:
    Uses the `GrB_PLUS_MONOID_T` monoid and the
    corresponding `GrB_FIRST_T` multiplicative operator:

    .. code-block:: C

        LAGraph_plus_first_int8
        LAGraph_plus_first_int16
        LAGraph_plus_first_int32
        LAGraph_plus_first_int64
        LAGraph_plus_first_uint8
        LAGraph_plus_first_uint16
        LAGraph_plus_first_uint32
        LAGraph_plus_first_uint64
        LAGraph_plus_first_fp32
        LAGraph_plus_first_fp64

- LAGraph_plus_second_T
    Uses the `GrB_PLUS_MONOID_T` monoid and the
    corresponding `GrB_SECOND_T` multiplicative operator:

    .. code-block:: C

        LAGraph_plus_second_int8
        LAGraph_plus_second_int16
        LAGraph_plus_second_int32
        LAGraph_plus_second_int64
        LAGraph_plus_second_uint8
        LAGraph_plus_second_uint16
        LAGraph_plus_second_uint32
        LAGraph_plus_second_uint64
        LAGraph_plus_second_fp32
        LAGraph_plus_second_fp64

- LAGraph_plus_one_T:
    Uses the `GrB_PLUS_MONOID_T` monoid and the
    corresponding `GrB_ONEB_T` multiplicative operator:

    .. code-block:: C

        LAGraph_plus_one_int8
        LAGraph_plus_one_int16
        LAGraph_plus_one_int32
        LAGraph_plus_one_int64
        LAGraph_plus_one_uint8
        LAGraph_plus_one_uint16
        LAGraph_plus_one_uint32
        LAGraph_plus_one_uint64
        LAGraph_plus_one_fp32
        LAGraph_plus_one_fp64

- LAGraph_any_one_T:
    Uses the `GrB_MIN_MONOID_T` for non-boolean types or
    `GrB_LOR_MONOID_BOOL` for boolean, and the `GrB_ONEB_T` multiplicative op.

    These semirings are very useful for unweighted graphs, or for algorithms
    that operate only on the sparsity structure of unweighted graphs:

    .. code-block:: C

        LAGraph_any_one_bool
        LAGraph_any_one_int8
        LAGraph_any_one_int16
        LAGraph_any_one_int32
        LAGraph_any_one_int64
        LAGraph_any_one_uint8
        LAGraph_any_one_uint16
        LAGraph_any_one_uint32
        LAGraph_any_one_uint64
        LAGraph_any_one_fp32
        LAGraph_any_one_fp64

