#file(GLOB_RECURSE SOURCES "src/*.cpp")
#file(GLOB_RECURSE TESTS "src/*.test.cpp")
#list(REMOVE_ITEM SOURCES ${TESTS})

set(SOURCES
        src/CircleMemoryPlan.cpp
        src/SimpleMemoryPlanner.cpp
        src/SimpleMemoryPlanner.h
        )

add_executable(circle_memory_plan "${SOURCES}")
target_include_directories(circle_memory_plan PRIVATE src)
#target_link_libraries(circle_memory_plan PRIVATE luci_interpreter)
target_link_libraries(circle_memory_plan foder)
target_link_libraries(circle_memory_plan nncc_common)
target_link_libraries(circle_memory_plan safemain)
target_link_libraries(circle_memory_plan oops)
target_link_libraries(circle_memory_plan hermes)
target_link_libraries(circle_memory_plan hermes_std)
target_link_libraries(circle_memory_plan loco)
target_link_libraries(circle_memory_plan mio_circle)
target_link_libraries(circle_memory_plan luci_env)
target_link_libraries(circle_memory_plan luci_import)
target_link_libraries(circle_memory_plan luci_service)
#target_link_libraries(circle2circle luci_pass)
target_link_libraries(circle_memory_plan luci_export)
target_link_libraries(circle_memory_plan arser)
target_link_libraries(circle_memory_plan vconone)

#target_link_libraries(circle_memory_plan PRIVATE oops)
#target_link_libraries(circle_memory_plan PRIVATE luci_lang)

#target_link_libraries(circle_memory_plan PRIVATE safemain)

install(TARGETS circle_memory_plan DESTINATION bin)

