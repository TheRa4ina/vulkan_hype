#Top-level

cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


project(vulkan_hype VERSION 1.0.1) #name of your project

find_package(Vulkan REQUIRED)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
find_package(VulkanHeaders)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

#Google test 
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/dddb219c3eb96d7f9200f09b0a381f016e6b4562.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

#shader compilation
add_subdirectory(shaders)

#myownstuff
add_subdirectory(src)
