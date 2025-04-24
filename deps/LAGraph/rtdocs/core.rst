LAGraph context and error handling
==================================

The sections below describe a set of functions that manage the LAGraph context
within a user application, and discuss how errors are handled.


LAGraph Context Functions
-------------------------

.. _lagraph_init:
.. doxygenfunction:: LAGraph_Init

.. _lagr_init:
.. doxygenfunction:: LAGr_Init

.. _lagraph_finalize:
.. doxygenfunction:: LAGraph_Finalize

.. doxygenfunction:: LAGraph_Version

.. doxygenfunction:: LAGraph_GetNumThreads

.. doxygenfunction:: LAGraph_SetNumThreads

Error handling
--------------

.. doxygendefine:: LAGRAPH_RETURN_VALUES

.. doxygendefine:: LAGRAPH_MSG_LEN

.. _lagraph_try:
.. doxygendefine:: LAGRAPH_TRY

.. _grb_try:
.. doxygendefine:: GRB_TRY
