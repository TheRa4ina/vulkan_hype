if(EXISTS "F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/libhello/HelloTests[1]_tests.cmake")
  include("F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/libhello/HelloTests[1]_tests.cmake")
else()
  add_test(HelloTests_NOT_BUILT HelloTests_NOT_BUILT)
endif()
