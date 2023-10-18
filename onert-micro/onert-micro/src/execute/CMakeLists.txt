message(STATUS "ONERT MICRO EXECUTE BUILD BEGIN")

include("${OM_PAL_DIR}/pal.cmake")

initialize_pal()

if (NOT PAL_INITIALIZED)
    message(STATUS "ERROR: PAL Failed to initialize, skip BUILD EXECUTE")
    return()
endif()

set(SOURCES
        OMKernelExecute.cpp
        )

# Add configure kernels
macro(REGISTER_KERNEL OPERATOR, NODE)
    list(APPEND SOURCES "kernels/${NODE}.cpp")
endmacro(REGISTER_KERNEL)

# To add REGISTER_KERNEL list
include(${KERNEL_REGISTER_FILE})

add_library(${OM_EXECUTE_LIB} STATIC ${SOURCES})

target_include_directories(${OM_EXECUTE_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_EXECUTE_LIB} PUBLIC ${OM_CORE_LIB})

target_include_directories(${OM_EXECUTE_LIB} PUBLIC ${LUCI_INTERPRETER_PAL_COMMON_DIR})
add_pal_to_target(${OM_EXECUTE_LIB})

message(STATUS "ONERT MICRO EXECUTE BUILD FINISHED")
