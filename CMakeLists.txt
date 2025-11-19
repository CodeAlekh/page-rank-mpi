cmake_minimum_required(VERSION 3.12)
project(mpi-page-rank LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

# Find MPI
find_package(MPI REQUIRED)
if (MPI_COMPILER_FLAGS)
  add_compile_options(${MPI_COMPILER_FLAGS})
endif()

include_directories(${MPI_INCLUDE_PATH})
set(EXTERNAL_LIBS ${MPI_LIBRARIES})
set(MPICXX "mpicxx" CACHE STRING "MPICXX command")
set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")

# Adds Compile Warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")

# Grab all .cpp files inside c/lib/
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS c/libs/*.cpp)
# For easy data/matrix file reading
add_compile_definitions(SRC_DIR="${CMAKE_SOURCE_DIR}")

# Create library with all lib/*.cpp
add_library(mylib ${LIB_SOURCES})

# Include directory
target_include_directories(mylib PUBLIC c/libs)

# Executable
add_executable(spmv_example c/src/spmv_example.cpp)
add_executable(page_rank c/src/page_rank.cpp)

# Link the library + MPI
target_link_libraries(spmv_example PRIVATE mylib ${MPI_LIBRARIES})
target_link_libraries(page_rank PRIVATE mylib ${MPI_LIBRARIES})