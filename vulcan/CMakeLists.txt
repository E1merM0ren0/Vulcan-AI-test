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
set(SHADER_SOURCES "")
file(GLOB SHADER_COMP_LIST CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
foreach(sh ${SHADER_COMP_LIST})
  get_filename_component(name ${sh} NAME_WE)
  list(APPEND SHADER_SOURCES ${sh})
endforeach()

add_custom_command(
  OUTPUT ${SPV_DIR}/.stamp
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPV_DIR}
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/matmul.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/rms_norm.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rms_norm.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/rms_norm_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rms_norm_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/quantize_absmax.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantize_absmax.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/quantize_ternary.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantize_ternary.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/scale_rows.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scale_rows.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/gelu.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gelu.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/gelu_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gelu_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/cross_entropy.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cross_entropy.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/cross_entropy_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cross_entropy_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/adam.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/adam.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/add_pos.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_pos.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/add_pos_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_pos_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/add.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/mul.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mul.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/mul_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mul_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/copy.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/zero.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/zero.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/embedding.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/embedding.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/embedding_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/embedding_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/reduce_sum_rows.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/reduce_sum_rows.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/softmax.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/softmax.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/softmax_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/softmax_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/split_heads.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/split_heads.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/merge_heads.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/merge_heads.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/layer_norm.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/layer_norm.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/layer_norm_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/layer_norm_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/tanh.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tanh.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/tanh_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tanh_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/silu.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/silu.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/silu_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/silu_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/relu.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/relu.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/relu_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/relu_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/mse_square.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mse_square.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/mse_bwd.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mse_bwd.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/add_bias.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_bias.comp
  COMMAND ${GLSLC} -fshader-stage=compute -O -o ${SPV_DIR}/rope.spv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rope.comp
  COMMAND ${CMAKE_COMMAND} -E touch ${SPV_DIR}/.stamp
  DEPENDS ${SHADER_SOURCES}
  VERBATIM)
add_custom_target(shaders DEPENDS ${SPV_DIR}/.stamp)

function(add_model name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE vulcan_engine)
  add_dependencies(${name} shaders)
endfunction()

add_model(vulcan_bitnet_gpt src/bitnet_gpt.cpp)
add_model(vulcan_bitnet src/bitnet.cpp)
add_model(vulcan_gpt src/gpt2.cpp)
