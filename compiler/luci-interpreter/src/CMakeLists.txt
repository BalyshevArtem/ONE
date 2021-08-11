nnas_find_package(TensorFlowSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowGEMMLowpSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowEigenSource EXACT 2.3.0 QUIET)
nnas_find_package(TensorFlowRuySource EXACT 2.3.0 QUIET)

if (NOT TensorFlowSource_FOUND)
  message(STATUS "Skipping luci-interpreter: TensorFlow not found")
  return()
endif ()

if (NOT TensorFlowGEMMLowpSource_FOUND)
  message(STATUS "Skipping luci-interpreter: gemmlowp not found")
  return()
endif ()

if (NOT TensorFlowEigenSource_FOUND)
  message(STATUS "Skipping luci-interpreter: Eigen not found")
  return()
endif ()

if (NOT TensorFlowRuySource_FOUND)
  message(STATUS "Skipping luci-interpreter: Ruy not found")
  return()
endif ()

add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(loader)

set(SOURCES
    "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/Interpreter.h"
    Interpreter.cpp "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/SimpleMemoryManager.h" SimpleMemoryManager.cpp
        "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/BuddyMemoryManager.h" BuddyMemoryManager.cpp)

add_library(luci_interpreter SHARED ${SOURCES})
target_include_directories(luci_interpreter PUBLIC "${LUCI_INTERPRETER_INCLUDE_DIR}")
target_include_directories(luci_interpreter PRIVATE "${LUCI_INTERPRETER_SOURCE_DIR}")
target_link_libraries(luci_interpreter
    PUBLIC luci_lang luci_interpreter_loader luci_interpreter_core
    PRIVATE nncc_common)

install(TARGETS luci_interpreter DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES BuddyMemoryManager.test.cpp)

GTest_AddTest(luci_interpreter_buddy_mmanager_test ${TEST_SOURCES})
target_link_libraries(luci_interpreter_buddy_mmanager_test luci_interpreter)