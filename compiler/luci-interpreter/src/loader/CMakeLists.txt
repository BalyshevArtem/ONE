set(SOURCES
    GraphLoader.h
    GraphLoader.cpp
    KernelBuilderHelper.h
    KernelBuilderHelper.cpp
    KernelBuilder.h
    KernelBuilder.cpp
    ModuleLoader.h
    ModuleLoader.cpp
    RuntimeToIR.h
    nodes/Builders.h
        CircleReader.h
        CircleReader.cpp)
        #${NNAS_PROJECT_SOURCE_DIR}/compiler/luci/import/include/luci/Import/CircleReader.h
        #${NNAS_PROJECT_SOURCE_DIR}/compiler/luci/import/src/CircleReader.cpp  )

# include kernel specific builders
macro(REGISTER_KERNEL NODE)
  #list(APPEND SOURCES "nodes/${NODE}.cpp")
endmacro(REGISTER_KERNEL)
include(${KERNEL_REGISTER_FILE})

add_library(${LUCI_INTERPRETER_LOADER} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_LOADER} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)
target_include_directories(${LUCI_INTERPRETER_LOADER} PUBLIC "${LUCI_INTERPRETER_PAL_DIR}")
target_include_directories(${LUCI_INTERPRETER_LOADER} PUBLIC "${LUCI_INTERPRETER_SOURCE_DIR}")


set(ONE_COMPILER_SRC_DIR "${NNAS_PROJECT_SOURCE_DIR}/compiler")
set(MICRO_ARM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../../")

target_link_libraries(${LUCI_INTERPRETER_LOADER} PUBLIC "${MICRO_ARM_BUILD_DIR}/mio-circle04/libmio_circle04.a")
target_link_libraries(${LUCI_INTERPRETER_LOADER} PUBLIC "${MICRO_ARM_BUILD_DIR}/mio-circle04/libmio_circle04_helper.a")

target_link_libraries(${LUCI_INTERPRETER_LOADER} PUBLIC mio_circle04)
target_link_libraries(${LUCI_INTERPRETER_LOADER} PUBLIC mio_circle04_helper)

target_link_libraries(${LUCI_INTERPRETER_LOADER}
        PUBLIC luci_lang ${LUCI_INTERPRETER_CORE}
        PRIVATE ${LUCI_INTERPRETER_KERNELS} nncc_common luci_plan)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES KernelBuilder.test.cpp)

GTest_AddTest(${LUCI_INTERPRETER_LOADER}_test ${TEST_SOURCES})
target_link_libraries(${LUCI_INTERPRETER_LOADER}_test ${LUCI_INTERPRETER_LOADER})
