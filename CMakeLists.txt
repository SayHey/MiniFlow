cmake_minimum_required(VERSION 3.15)

project(MiniFlow CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# set up tests
option(ENABLE_TESTING "Enable Test Builds" ON)
if(ENABLE_TESTING)
    enable_testing()
    message("Building Tests")

    include(GoogleTest)
    add_executable(tensor_test test/tensor_test.cpp)
    target_link_libraries(tensor_test PRIVATE CONAN_PKG::gtest)
    gtest_add_tests(TARGET tensor_test
                    TEST_SUFFIX .noArgs
                    TEST_LIST   noArgsTests
    )
endif()

#
# add_executable(nn src/nn.cpp)