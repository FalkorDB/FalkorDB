/*
 * Copyright (c) 2019,2020 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#include "GB_jit_cache.h"
#include "GraphBLAS.h"
// from GraphBLAS.h (for example):
// #define GxB_IMPLEMENTATION_MAJOR 6
// #define GxB_IMPLEMENTATION_MINOR 0
// #define GxB_IMPLEMENTATION_SUB   3

namespace jit {

// Get the directory in home to use for storing the cache
    std::string get_user_home_cache_dir() {
        auto home_dir = std::getenv("HOME");
        if (home_dir != nullptr) {
            std::string Major_ver = GB_XSTR (GxB_IMPLEMENTATION_MAJOR) ;
            std::string Minor_ver = GB_XSTR (GxB_IMPLEMENTATION_MINOR) ;
            std::string Imple_sub = GB_XSTR (GxB_IMPLEMENTATION_SUB) ;
            return std::string(home_dir) + "/.SuiteSparse/GrB"
                   + Major_ver+"."+Minor_ver+"."+Imple_sub;
        } else {
            return std::string();
        }
    }

// Get the directory in home to use for storing the cache
    std::string get_user_graphblas_source_path() {
        auto gb_home = std::getenv("GRAPHBLAS_SOURCE_PATH");
        if (gb_home != nullptr) return std::string(gb_home);
        else return std::string();
    }


// Default `GRAPHBLAS_CACHE_PATH` to `$HOME/.GraphBLAS`.
// This definition can be overridden at compile time by specifying a
// `-DGRAPHBLAS_CACHE_PATH=/kernel/cache/path` CMake argument.
// This path is used in the `getCacheDir()` function below.
#if !defined(GRAPHBLAS_CACHE_PATH)
#define GRAPHBLAS_CACHE_PATH get_user_home_cache_dir() 
#endif

/**
 * @brief Get the string path to the JITIFY kernel cache directory.
 *
 * This path can be overridden at runtime by defining an environment variable
 * named `GRAPHBLAS_CACHE_PATH`. The value of this variable must be a path
 * under which the process' user has read/write priveleges.
 *
 * This function returns a path to the cache directory, creating it if it
 * doesn't exist.
 *
 * The default cache directory is `$HOME/.GraphBLAS`. If no overrides
 * are used and if $HOME is not defined, returns an empty path and file 
 * caching is not used.
 **/
std::string getCacheDir() {
  // The environment variable always overrides the
  // default/compile-time value of `GRAPHBLAS_CACHE_PATH`
  // FIXME: use GB_jitifyer_get_cache_path ( ) here.
  //std::cout<<"Cache-dir runtime=>"<< get_user_home_cache_dir() << std::endl;
  auto kernel_cache_path_env = std::getenv("GRAPHBLAS_CACHE_PATH");
  //auto kernel_cache_path_env = get_user_home_cache_dir();
  auto kernel_cache_path = (kernel_cache_path_env != nullptr ? kernel_cache_path_env
                                       : GRAPHBLAS_CACHE_PATH);

  struct stat st;
  //if ( (stat( kernel_cache_path.c_str(), &st) != 0) ) {
  if ( (stat( kernel_cache_path.c_str(), &st) != 0) ) {
    // `mkdir -p` the kernel cache path if it doesn't exist
//    printf("cache is going to path %s\n", kernel_cache_path.c_str());
    int status;
    //status = std::filesystem::create_directories(kernel_cache_path.c_str());
    status = std::filesystem::create_directories(kernel_cache_path);
//    if (status != 0 ) return std::string();
    //boost::filesystem::create_directories(kernel_cache_path);
  }
  return std::string(kernel_cache_path);
}

GBJitCache::GBJitCache() { }

GBJitCache::~GBJitCache() { }


//void GBJitCache::macrofy() {
//    printf("GOT HERE and shouldn't have!\n");
//}


std::mutex GBJitCache::_kernel_cache_mutex;
std::mutex GBJitCache::_program_cache_mutex;

std::string GBJitCache::getFile(
    File_Desc &file_object )
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);

    // Macrofied version
    auto cached_file = getCachedFile( file_object, file_map );
    return *std::get<1>( cached_file ).get();
}

named_prog<jitify::experimental::Program> GBJitCache::getProgram(
    std::string const& prog_name, 
    std::string const& cuda_source,
    std::vector<std::string> const& given_headers,
    std::vector<std::string> const& given_options,
    jitify::experimental::file_callback_type file_callback)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);
//    printf(" jit_cache get program %s\n", prog_name.c_str());

    return getCached(prog_name, program_map, 
        [&](){
            return jitify::experimental::Program(cuda_source,
                                        given_headers,
                                        given_options,
                                        file_callback);
        }
    );
}

named_prog<jitify::experimental::KernelInstantiation> GBJitCache::getKernelInstantiation(
    std::string const& kern_name,
    named_prog<jitify::experimental::Program> const& named_program,
    std::vector<std::string> const& arguments)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_kernel_cache_mutex);

    std::string prog_name = std::get<0>(named_program);
    jitify::experimental::Program& program = *std::get<1>(named_program);

    // Make instance name e.g. "prog_binop.kernel_v_v_int_int_long int_Add"
    std::string kern_inst_name = kern_name;
    for ( auto&& arg : arguments ) kern_inst_name += '_' + arg;

    //printf(" got kernel instance %s\n",kern_inst_name.c_str());

    return getCached(kern_inst_name, kernel_inst_map, 
        [&](){return program.kernel(kern_name)
                            .instantiate(arguments);
        }
    );
}

// Another overload for getKernelInstantiation which might be useful to get
// kernel instantiations in one step
// ------------------------------------------------------------------------
/*
jitify::experimental::KernelInstantiation GBJitCache::getKernelInstantiation(
    std::string const& kern_name,
    std::string const& prog_name,
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    file_callback_type file_callback = nullptr)
{
    auto program = getProgram(prog_name,
                              cuda_source,
                              given_headers,
                              given_options,
                              file_callback);
    return getKernelInstantiation(kern_name, program);
}
*/

GBJitCache::cacheFile::cacheFile(std::string file_name)
 : _file_name{file_name}
{ }

GBJitCache::cacheFile::~cacheFile() { }

std::string GBJitCache::cacheFile::read_file()
{
    // Open file (duh)
    int fd = open ( _file_name.c_str(), O_RDWR );
    if ( fd == -1 ) {
        // TODO: connect errors to GrB_error result
//        printf(" failed to open cache file %s\n",_file_name.c_str());
        successful_read = false;
        return std::string();
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_read = false;
        return std::string();
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "rb" );

    // Get file length
    fseek( fp , 0L , SEEK_END);
    size_t file_size = ftell( fp );
    rewind( fp );

    // Allocate memory of file length size
    std::string content;
    content.resize(file_size);

    char *buffer = content.data();

    // Copy file into buffer
    if( fread(buffer, file_size, 1, fp) != 1 ) {
        //printf(" failed to read cache file %s\n",_file_name.c_str());
        successful_read = false;
        fclose(fp);
//        free(buffer); FIXME: Shouldn't need to free buffer since it's RAII
        return content; // FIXME: use unique_ptr here
    }

//    printf("about to close\n");
    fclose(fp);
    successful_read = true;
//    printf(" read cache file %s\n",_file_name.c_str());

    return content;
}

void GBJitCache::cacheFile::write(std::string content)
{
    // Open file and create if it doesn't exist, with access 0600
    int fd = open ( _file_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if ( fd == -1 ) {
        //printf(" failed to open cache file for write %s\n",_file_name.c_str());
        successful_write = false;
        return;
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_write = false;
        return;
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "wb" );

    // Copy string into file
    if( fwrite(content.c_str(), content.length(), 1, fp) != 1 ) {
        //printf(" failed to write cache file %s\n",_file_name.c_str());
        successful_write = false;
        fclose(fp);
        return;
    }
    fclose(fp);

    successful_write = true;
    //printf(" wrote cache file %s\n",_file_name.c_str());
    
    return;
}

} // namespace jit
