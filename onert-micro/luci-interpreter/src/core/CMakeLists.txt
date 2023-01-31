set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/DataType.h"
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/core/Tensor.h"
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp)

#macro(REGISTER_KERNEL OPERATOR, NODE)
#    #list(APPEND SOURCES "${NODE}.h")
#    list(APPEND SOURCES "../kernels/${NODE}.cpp")
#endmacro(REGISTER_KERNEL)

add_library(${LUCI_INTERPRETER_CORE} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
    set_target_properties(${LUCI_INTERPRETER_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

add_subdirectory(reader)

target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC "luci_micro_circle_reader${READER_SUFFIX}")
#target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC ${LUCI_INTERPRETER_KERNELS})
target_link_libraries(${LUCI_INTERPRETER_CORE} PUBLIC luci_micro_circle_schema)
#if (LUCI_INTERPRETER_KERNELS_BUILD_LIST)
#    target_compile_definitions(${LUCI_INTERPRETER_LOADER} PRIVATE USE_GENERATED_LIST=1)
#endif(LUCI_INTERPRETER_KERNELS_BUILD_LIST)

target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")
#target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_PAL_DIR}")
#target_include_directories(${LUCI_INTERPRETER_CORE} PUBLIC "${LUCI_INTERPRETER_KERNELS}")
