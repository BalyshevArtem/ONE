message(STATUS "ONERT MICRO IMPORT BUILD BEGIN")

set(SOURCES
        OMGraphLoader.cpp
        OMExecutionPlanCreator.cpp
        OMKernelConfiguration.cpp
        OMKernelConfigureBuilder.cpp
        kernels/custom_gru.cpp
        )

# Add configure kernels
macro(REGISTER_KERNEL OPERATOR, NODE)
    list(APPEND SOURCES "kernels/${NODE}.cpp")
endmacro(REGISTER_KERNEL)

# To add REGISTER_KERNEL list
include(${KERNEL_REGISTER_FILE})

add_library(${OM_IMPORT_LIB} STATIC ${SOURCES})

target_include_directories(${OM_IMPORT_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_IMPORT_LIB} PUBLIC ${OM_CORE_LIB})

message(STATUS "ONERT MICRO IMPORT BUILD FINISHED")
