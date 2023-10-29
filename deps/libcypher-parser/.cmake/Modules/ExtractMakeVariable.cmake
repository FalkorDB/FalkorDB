function (extract_make_variable filename varname outvar prefix)
  file (STRINGS ${filename} contents)
  foreach (line IN LISTS contents)
    if (line MATCHES "^${varname}*")
      list (REMOVE_AT line 0)
      foreach (item IN LISTS line)
        string (STRIP ${item} item)
        list (APPEND outvar ${prefix}${item})
      endforeach (item)
    endif (line MATCHES "^${varname}*")
  endforeach (line)
  set(${outvar} PARENT_SCOPE)
endfunction (extract_make_variable)