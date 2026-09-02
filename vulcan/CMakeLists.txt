cmake_minimum_required(VERSION 3.20)
project(vulcan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

set(ENGINE_SOURCES
  src/vulkan_api.cpp
  src/context.cpp
  src/graph.cpp
  src/gguf.cpp)

set(ENGINE_HEADERS
  src/vulkan_api.h
  src/context.h
  src/graph.h
  src/gguf.h)

add_library(vulcan_engine STATIC ${ENGINE_SOURCES})
target_include_directories(vulcan_engine PUBLIC src)
target_link_libraries(vulcan_engine PUBLIC Vulkan::Vulkan)

set(SPV_DIR ${CMAKE_BINARY_DIR}/spv)

# Auto-compile every .comp in shaders/ -> <name>.spv. New shaders are picked up
# automatically (no manual glslc lines needed). Globs are CONFIGURE_DEPENDS so
# re-running build picks up added/removed files.
file(GLOB SHADER_COMP_LIST CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
set(SHADER_SPV_LIST "")
foreach(sh ${SHADER_COMP_LIST})
  get_filename_component(name ${sh} NAME_WE)
  set(spv ${SPV_DIR}/${name}.spv)
  add_custom_command(
    OUTPUT ${spv}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SPV_DIR}
    COMMAND ${GLSLC} -fshader-stage=compute -O -o ${spv} ${sh}
    DEPENDS ${sh}
    VERBATIM)
  list(APPEND SHADER_SPV_LIST ${spv})
endforeach()
add_custom_target(shaders ALL DEPENDS ${SHADER_SPV_LIST})

function(add_model name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE vulcan_engine)
  add_dependencies(${name} shaders)
endfunction()

add_model(vulcan_bitnet_gpt src/bitnet_gpt.cpp)
add_model(vulcan_bitnet src/bitnet.cpp)
add_model(vulcan_gpt src/gpt2.cpp)
