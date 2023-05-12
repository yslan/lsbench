include(FetchContent)
include(ExternalProject)

set(GINKGO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})

FetchContent_Declare(
  ginkgo
  GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
  GIT_TAG develop
)
FetchContent_GetProperties(ginkgo)
if (NOT ginkgo_POPULATED)
  FetchContent_Populate(ginkgo)
endif()

ExternalProject_Add(ginkgo_ext
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${ginkgo_SOURCE_DIR}
  CMAKE_ARGS -Dhiprand_DIR=/opt/rocm-5.3.0/lib/cmake/hiprand/
  -Drocrand_DIR=/opt/rocm-5.3.0/lib/cmake/rocrand/
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
#target_link_libraries(lsbench PRIVATE Ginkgo::ginkgo)
