add_test([=[helloClass.helloStuff_GivingOne_ReturnsOne]=]  F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/libhello/HelloTests.exe [==[--gtest_filter=helloClass.helloStuff_GivingOne_ReturnsOne]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[helloClass.helloStuff_GivingOne_ReturnsOne]=]  PROPERTIES WORKING_DIRECTORY F:/0main_stuff/prog_stuff/projects/cpp/vulkan_hype/build/x64-Debug/libhello SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  HelloTests_TESTS helloClass.helloStuff_GivingOne_ReturnsOne)
