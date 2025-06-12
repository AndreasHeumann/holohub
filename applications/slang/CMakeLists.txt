#
# Place the license header here
#

cmake_minimum_required(VERSION 3.20)

project(slang)

find_package(holoscan 3.3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_executable(slang
    src/main.cpp
)

target_link_libraries(slang
  PRIVATE
    holoscan::core
    holoscan::ops::slang_shader
)

# Copy shader file
add_custom_target(slang_shader_file
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/shader.slang" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/shader.slang"
)
add_dependencies(slang slang_shader_file)


if(BUILD_TESTING)
  add_test(slang-test
    COMMAND echo "Add test coverage for slang here"
  )
endif()

