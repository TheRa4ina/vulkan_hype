# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-src"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-build"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
