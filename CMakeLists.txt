cmake_minimum_required(VERSION 3.16)
project(WSJoin)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-pie")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

# Find required packages
find_package(PkgConfig)

# Try to find jsoncpp, but make it optional
if(PkgConfig_FOUND)
    pkg_check_modules(JSONCPP jsoncpp)
endif()

# If jsoncpp not found via pkg-config, try manual find
if(NOT JSONCPP_FOUND)
    find_library(JSONCPP_LIBRARY NAMES jsoncpp)
    find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h)

    if(JSONCPP_LIBRARY AND JSONCPP_INCLUDE_DIR)
        set(JSONCPP_FOUND TRUE)
        set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARY})
        set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIR})
    endif()
endif()

# If still not found, try common locations
if(NOT JSONCPP_FOUND)
    find_library(JSONCPP_LIBRARY NAMES jsoncpp PATHS /usr/lib/x86_64-linux-gnu)
    if(JSONCPP_LIBRARY)
        set(JSONCPP_FOUND TRUE)
        set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARY})
        # Try common include paths
        find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h
                  PATHS /usr/include /usr/local/include /usr/include/jsoncpp)
        if(JSONCPP_INCLUDE_DIR)
            set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIR})
        endif()
    endif()
endif()

# Add executables
add_executable(naive_join naive_join.cpp)
add_executable(ws_join ws_join.cpp)

# Include directories (if jsoncpp found)
if(JSONCPP_FOUND)
    if(JSONCPP_INCLUDE_DIRS)
        include_directories(${JSONCPP_INCLUDE_DIRS})
    endif()
    target_link_libraries(naive_join ${JSONCPP_LIBRARIES})
    target_link_libraries(ws_join ${JSONCPP_LIBRARIES})
    if(JSONCPP_CFLAGS_OTHER)
        target_compile_options(naive_join PRIVATE ${JSONCPP_CFLAGS_OTHER})
        target_compile_options(ws_join PRIVATE ${JSONCPP_CFLAGS_OTHER})
    endif()
    message(STATUS "jsoncpp found and will be used")
    message(STATUS "JSONCPP_LIBRARIES: ${JSONCPP_LIBRARIES}")
    message(STATUS "JSONCPP_INCLUDE_DIRS: ${JSONCPP_INCLUDE_DIRS}")
else()
    message(WARNING "jsoncpp not found - you may need to install libjsoncpp-dev")
    message(STATUS "You can install it with: sudo apt install libjsoncpp-dev")
    message(STATUS "JSONCPP not found - compilation may fail")
endif()


# Set compiler flags for optimization
target_compile_options(naive_join PRIVATE -O3 -march=native)
target_compile_options(ws_join PRIVATE -O3 -march=native)


# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(naive_join Threads::Threads)
target_link_libraries(ws_join Threads::Threads)
