message(STATUS "ONERT MICRO OPTIMIZER BUILD BEGIN")

set(SOURCES
        OMOptimizer.cpp
        pass/OMFindInplaceOpPass.cpp
        )

# TODO add list for passes
add_library(${OM_OPTIMIZE_LIB} STATIC ${SOURCES})

target_include_directories(${OM_OPTIMIZE_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_OPTIMIZE_LIB} PUBLIC ${OM_CORE_LIB})

message(STATUS "ONERT MICRO OPTIMIZER BUILD FINISHED")
