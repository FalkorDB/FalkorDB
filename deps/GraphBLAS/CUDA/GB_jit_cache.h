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

#ifndef GB_JIT_CACHE_H_
#define GB_JIT_CACHE_H_

#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>


#define JITIFY_USE_CACHE 1

namespace jit {

std::string get_user_home_cache_dir();
std::string get_user_graphblas_source_path();
std::string getCacheDir(void);

template <typename Tv>
using named_prog = std::pair<std::string, std::shared_ptr<Tv>>;

// Basic file descriptor to enable file manipulation with caching 
class File_Desc
{
public:
   virtual void open( const char *path_and_file, const char *mode) {}
   virtual void close() {}
   virtual void macrofy() {

       printf("Uh oh. this isn't good\n");

   }
   std::string filename;
};

/**
 * @brief Get the string path to the JITIFY kernel cache directory.
 *
 * This path can be overridden at runtime by defining an environment variable
 * named `GB_CUDA_KERNEL_CACHE_PATH`. The value of this variable must be a path
 * under which the process' user has read/write priveleges.
 *
 * This function returns a path to the cache directory, creating it if it
 * doesn't exist.
 *
 * The default cache directory `~/.GraphBLAS_kernel_cache`.
 **/

class GBJitCache
{
public:

    /**---------------------------------------------------------------------------*
     * @brief Get a process wide singleton cache object
     * 
     *---------------------------------------------------------------------------**/
    static GBJitCache& Instance() {
        // Meyers' singleton is thread safe in C++11
        // Link: https://stackoverflow.com/a/1661564
        static GBJitCache cache;
        return cache;
    }

    GBJitCache();
    ~GBJitCache();

    /**---------------------------------------------------------------------------*
     * @brief Get the file object
     * 
     * Searches an internal in-memory cache and file based cache for the file 
     * and if not found, opens the file, calls macrofy, closes the file 
     * 
     * @param file_desc [in] object representing file:  open, macrofy, close 
     * @return  string name of file, or 'error' if not able to create file  
     *---------------------------------------------------------------------------**/
    std::string getFile( File_Desc & file_obj );

    /**---------------------------------------------------------------------------*
     * @brief Get the Kernel Instantiation object
     * 
     * Searches an internal in-memory cache and file based cache for the kernel
     * and if not found, JIT compiles and returns the kernel
     * 
     * @param kern_name [in] name of kernel to return
     * @param program   [in] Jitify preprocessed program to get the kernel from
     * @param arguments [in] template arguments for kernel in vector of strings
     * @return  Pair of string kernel identifier and compiled kernel object
     *---------------------------------------------------------------------------**/
    named_prog<jitify::experimental::KernelInstantiation> getKernelInstantiation(
        std::string const& kern_name,
        named_prog<jitify::experimental::Program> const& program,
        std::vector<std::string> const& arguments);

    /**---------------------------------------------------------------------------*
     * @brief Get the Jitify preprocessed Program object
     * 
     * Searches an internal in-memory cache and file based cache for the Jitify
     * pre-processed program and if not found, JIT processes and returns it
     * 
     * @param prog_file_name [in] name of program to return
     * @param cuda_source    [in] string source code of program to compile
     * @param given_headers  [in] vector of strings representing source or names of
     *  each header included in cuda_source
     * @param given_options  [in] vector of strings options to pass to NVRTC
     * @param file_callback  [in] pointer to callback function to call whenever a
     *  header needs to be loaded
     * @return named_prog<jitify::experimental::Program> 
     *---------------------------------------------------------------------------**/
    named_prog<jitify::experimental::Program> getProgram(
        std::string const& prog_file_name, 
        std::string const& cuda_source = "",
        std::vector<std::string> const& given_headers = {},
        std::vector<std::string> const& given_options = {},
        jitify::experimental::file_callback_type file_callback = nullptr);

private:
    template <typename Tv>
    using umap_str_shptr = std::unordered_map<std::string, std::shared_ptr<Tv>>;

    umap_str_shptr<std::string>                                file_map;
    umap_str_shptr<jitify::experimental::KernelInstantiation>  kernel_inst_map;
    umap_str_shptr<jitify::experimental::Program>              program_map;

    /*
    Even though this class can be used as a non-singleton, the file cache
    access should remain limited to one thread per process. The lockf locks can
    prevent multiple processes from accessing the file but are ineffective in
    preventing multiple threads from doing so as the lock is shared by the
    entire process.
    Therefore the mutexes are static.
    */
    static std::mutex _file_cache_mutex;
    static std::mutex _kernel_cache_mutex;
    static std::mutex _program_cache_mutex;

private:
    /**---------------------------------------------------------------------------*
     * @brief Class to allow process wise exclusive access to cache files
     *
     *---------------------------------------------------------------------------**/
    class cacheFile
    {
    private:
        std::string _file_name ;
        // FIXME this isn't used, is it?
        std::string _dir_name = "~/.GraphBLAS_kernel_cache/";   // FIXME
        bool successful_read = false;
        bool successful_write = false;
    public:
        cacheFile(std::string file_name);
        ~cacheFile();

        /**---------------------------------------------------------------------------*
         * @brief Read this file and return the contents as a std::string
         * 
         *---------------------------------------------------------------------------**/
        std::string read_file();

        /**---------------------------------------------------------------------------*
         * @brief Write the passed string to this file
         * 
         *---------------------------------------------------------------------------**/
        void write(std::string);

        /**---------------------------------------------------------------------------*
         * @brief Check whether the read() operation on the file completed successfully
         * 
         * @return true Read was successful. String returned by `read()` is valid
         * @return false Read was unsuccessful. String returned by `read()` is empty
         *---------------------------------------------------------------------------**/
        bool is_read_successful() { return successful_read; }

        /**---------------------------------------------------------------------------*
         * @brief Check whether the write() operation on the file completed successfully
         * 
         * @return true Write was successful.
         * @return false Write was unsuccessful. File state is undefined
         *---------------------------------------------------------------------------**/
        bool is_write_successful() { return successful_write; }
    };

private:

    template <typename T, typename FileDescType>
    named_prog<T> getCachedFile(
            FileDescType &file_object,
        umap_str_shptr<T>& map )
    {

//        printf("INside get cached file\n");
        std::string name = file_object.filename;

        // Find memory cached T object
        auto it = map.find(name);
        if ( it != map.end()) {
//            std::cout<<"found memory-cached file "<<name<<std::endl;
            return std::make_pair(name, it->second);
        }
        else { // Find file cached T object
            bool successful_read = false;
            std::string serialized;
            std::string cache_dir = getCacheDir();
            std::string file_name = cache_dir + "/" + name;
            if (not cache_dir.empty() ) {
                // TODO: Use OS-agnostic path separator here
//                std::cout<<"looking for prog in file "<<file_name<<std::endl;
                file_object.open(file_name.c_str(), "r");
                cacheFile file{file_name};
                serialized = file.read_file();
                successful_read = file.is_read_successful();
//                std::cout << "successful_read: " << successful_read << std::endl;
                if(successful_read) {
                    file_object.close();
//                    std::cout << "Just closed" << std::endl;
                }
            }
            if (not successful_read) {
                // JIT compile and write to file if possible
//                std::cout << "not successful read. macrofying" << std::endl;
                file_object.open(file_name.c_str(), "w");
                file_object.macrofy();
//                std::cout<<" got fresh content for "<<name<<std::endl;
                file_object.close();

                if (not cache_dir.empty()) {
//                    std::cout<<"writing in file "<<file_name<<std::endl;
                    cacheFile file{file_name};

                    cacheFile macrofied{name};
                    serialized = macrofied.read_file();
                    file.write(serialized);
                }
            }
            // Add deserialized T to cache and return
            map[name] = std::make_shared<std::string>(serialized);
            //std::cout<<"storing file in memory "<<name<<std::endl;
            return std::make_pair(name, map[name]);
        }
    }


    // GetCached 
    // This does a lot. 1) it checks if the file named is in the memory cache 
    //                  2) checks the disk cache 
    //                  3) compiles and caches the result and returns the program
    template <typename T, typename FallbackFunc>
    named_prog<T> getCached(
        std::string const& name,
        umap_str_shptr<T>& map,
        FallbackFunc func) {

        // Find memory cached T object
        auto it = map.find(name);
        if ( it != map.end()) {
//            std::cout<<"found memory-cached prog "<<name<<std::endl;
            return std::make_pair(name, it->second);
        }
        else { // Find file cached T object
            bool successful_read = false;
            std::string serialized;
            #if defined(JITIFY_USE_CACHE)
                std::string cache_dir = getCacheDir();
                if (not cache_dir.empty() ) {
                    // TODO: Use OS-agnostic path separator
                    std::string file_name = cache_dir + "/" + name;
                    //std::cout<<"looking for prog in file "<<file_name<<std::endl;

                    cacheFile file{file_name};
                    serialized = file.read_file();
                    successful_read = file.is_read_successful();
                }
            #endif
            if (not successful_read) {
                // JIT compile and write to file if possible
//                    std::cout << "compiling now" << std::endl;
                auto f = func();

//                    std::cout << "completed func()" << std::endl;
                serialized = f.serialize();
//                std::cout<<" compiled serialized prog "<<name<<std::endl;

                #if defined(JITIFY_USE_CACHE)
                    if (not cache_dir.empty()) {
                        std::string file_name = cache_dir + "/" + name;
//                        std::cout<<"writing prog in file "<<file_name<<std::endl;
                        cacheFile file{file_name};
                        file.write(serialized);
                    }
                #endif
            }
            // Add deserialized T to cache and return
            auto program = std::make_shared<T>(T::deserialize(serialized));
            map[name] = program;
            //std::cout<<"storing prog in memory "<<name<<std::endl;
            return std::make_pair(name, program);
        }
    }
};

} // namespace jit


#endif // GB_JIT_CACHE_H_
