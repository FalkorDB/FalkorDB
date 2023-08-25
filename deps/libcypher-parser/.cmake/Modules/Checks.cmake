check_include_file (dlfcn.h HAVE_DLFCN_H)
check_include_file (endian.h HAVE_ENDIAN_H)
check_include_file (inttypes.h HAVE_INTTYPES_H)
check_include_file (libkern/OSByteOrder.h HAVE_LIBKERN_OSBYTEORDER_H)
check_include_file (memory.h HAVE_MEMORY_H)
check_include_file (stdbool.h HAVE_STDBOOL_H)
check_include_file (stdint.h HAVE_STDINT_H)
check_include_file (stdlib.h HAVE_STDLIB_H)
check_include_file (strings.h HAVE_STRINGS_H)
check_include_file (string.h HAVE_STRING_H)
check_include_file (sys/endian.h HAVE_SYS_ENDIAN_H)
check_include_file (sys/stat.h HAVE_SYS_STAT_H)
check_include_file (sys/types.h HAVE_SYS_TYPES_H)
check_include_file (unistd.h HAVE_UNISTD_H)
check_include_file (stddef.h HAVE_STDDEF_H)

check_symbol_exists (strerror_r string.h HAVE_STRERROR_R)
check_symbol_exists (open_memstream stdio.h HAVE_OPEN_MEMSTREAM)

if (HAVE_STDLIB_H AND HAVE_STDDEF_H)
 set (STDC_HEADERS 1)
endif ()

if (HAVE_STRERROR_R)
  set (HAVE_DECL_STRERROR_R 1)
endif (HAVE_STRERROR_R)

find_package (Threads)
set (HAVE_PTHREADS ${CMAKE_USE_PTHREADS_INIT})

set (LT_OBJDIR ".libs/")
set (NDEBUG 1)

check_c_source_compiles ("void fun(int n) { 
  int arr[n];
}  
int main() { 
   fun(6);
   return 0;
}" HAVE_C_VARARRAYS )

check_c_source_compiles ("#include <string.h>
int main() {
  char * c;
  c = strerror_r(0,c,0);
  return 0;
}" STRERROR_R_CHAR_P )

check_c_source_compiles ("int main() {
  _Bool b = 1;
  return 0;
}" HAVE__BOOL )

check_c_source_compiles ("#include <sys/types.h>
int main() {
  size_t b = 1;
  return 0;
}" HAVE_SIZE_T )

if (NOT HAVE_SIZE_T)
 set (size_t "unsigned int")
endif ()

check_c_source_compiles ("#include <sys/types.h>
int main() {
  ssize_t b = 1;
  return 0;
}" HAVE_SSIZE_T )

if (NOT HAVE_SSIZE_T)
 set (ssize_t "int")
endif ()

check_c_source_compiles ("#include <assert.h>
static_assert(sizeof(int) == 4, \"Code relies on int being exactly 4 bytes\");
int main(void) {
    return 0;
}" HAVE_C_STATIC_ASSERT)

if (NOT HAVE_C_STATIC_ASSERT)
  set ("static_assert" "typedef void _no_static_assert")
endif ()

foreach (KEYWORD "inline" "__inline__" "__inline")
  if (NOT DEFINED C_INLINE)
    set (CMAKE_REQUIRED_DEFINITIONS "-Dinline=${KEYWORD}")

    check_c_source_compiles ("typedef int foo_t;
      static inline foo_t static_foo(){return 0;}
      foo_t foo(){return 0;}
      int main(int argc, char *argv[]){return 0;}" 
      C_HAS_${KEYWORD} )
    set (CMAKE_REQUIRED_DEFINITIONS)

    if (C_HAS_${KEYWORD})
      set (C_INLINE TRUE)
      if (NOT C_HAS_inline)
        set (inline ${KEYWORD})
      endif ()
    endif (C_HAS_${KEYWORD})
  endif (NOT DEFINED C_INLINE)
endforeach (KEYWORD)