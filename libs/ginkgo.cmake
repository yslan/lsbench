include(ExternalProject)

set(GINKGO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(GINKGO_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib64")
set(GINKGO_INCDIR "${CMAKE_INSTALL_PREFIX}/include")

ExternalProject_Add(ginkgo_ext
  GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
  GIT_TAG develop
  CMAKE_ARGS -Dhiprand_DIR=$ENV{ROCM_PATH}/lib/cmake/hiprand/
    -Drocrand_DIR=$ENV{ROCM_PATH}/lib/cmake/rocrand/
    -DCMAKE_INSTALL_PREFIX=${GINKGO_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DGINKGO_BUILD_MPI=OFF
    -DGINKGO_BUILD_HWLOC=OFF
    -DGINKGO_BUILD_CUDA=OFF
    -DGINKGO_BUILD_OMP=OFF
    -DGINKGO_BUILD_HIP=ON
)

add_dependencies(lsbench ginkgo_ext)
target_link_libraries(lsbench PRIVATE
  ${GINKGO_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ginkgo${CMAKE_SHARED_LIBRARY_SUFFIX})
target_include_directories(lsbench PRIVATE ${GINKGO_INCDIR})
