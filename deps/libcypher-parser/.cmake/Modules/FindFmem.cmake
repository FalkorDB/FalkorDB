# - Find fmem
# Find the native fmem headers and libraries.
#
# FMEM_INCLUDE_DIRS	- where to find fmem.h, etc.
# FMEM_LIBRARIES	- List of libraries when using fmem.
# FMEM_FOUND	- True if fmem has been found.

# Look for the header file.
find_path (FMEM_INCLUDE_DIR fmem.h)

# Look for the library.
find_library (FMEM_LIBRARY NAMES fmem)

# Handle the QUIETLY and REQUIRED arguments and set FMEM_FOUND to TRUE if all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FMEM DEFAULT_MSG FMEM_LIBRARY FMEM_INCLUDE_DIR)

# Copy the results to the output variables.
if (FMEM_FOUND)
    set (FMEM_LIBRARIES ${FMEM_LIBRARY})
    set (FMEM_INCLUDE_DIRS ${FMEM_INCLUDE_DIR})
else (FMEM_FOUND)
    set (FMEM_LIBRARIES)
    set (FMEM_INCLUDE_DIRS)
endif (FMEM_FOUND)

mark_as_advanced (FMEM_INCLUDE_DIRS FMEM_LIBRARIES)
