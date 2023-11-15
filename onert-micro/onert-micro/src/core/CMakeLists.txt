message(STATUS "ONERT MICRO CORE BUILD BEGIN")

##build import lib
#add_subdirectory(${OM_SOURCE_IMPORT_DIR})
##build execute lib
#add_subdirectory(${OM_SOURCE_EXECUTE_DIR})
##build optimize lib
#add_subdirectory(${OM_SOURCE_OPTIMIZE_DIR})

set(SOURCES
        OMKernel.cpp
        OMKernelType.cpp
        OMRuntimeContext.cpp
        OMRuntimeStorage.cpp
        OMRuntimeGraph.cpp
        OMRuntimeModule.cpp
        OMShape.cpp
        memory/OMMemoryManager.cpp
        memory/OMRuntimeAllocator.cpp
        reader/OMCircleReader.cpp)

add_library(${OM_CORE_LIB} STATIC ${SOURCES})

target_include_directories(${OM_CORE_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_include_directories(${OM_CORE_LIB} PUBLIC "${GENERATED_INCLUDE_DIR}")

#target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_CIRCLE_SCHEMA})
#target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_IMPORT_LIB})
#target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_EXECUTE_LIB})
#target_link_libraries(${OM_CORE_LIB} PUBLIC ${OM_OPTIMIZE_LIB})

message(STATUS "ONERT MICRO CORE BUILD FINISHED")
