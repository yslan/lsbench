include(FetchContent)
include(ExternalProject)

set(ROCALUTION_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(ROCALUTION_LIBDIR ${ROCALUTION_INSTALL_DIR}/lib)
set(ROCALUTION_INCDIR ${ROCALUTION_INSTALL_DIR}/include)


FetchContent_Declare(
  rocalution
  GIT_REPOSITORY https://github.com/thilinarmtb/rocALUTION
  GIT_TAG fixes_for_rocm_5_3_0
)
FetchContent_GetProperties(rocalution)
if (NOT rocalution_POPULATED)
  FetchContent_Populate(rocalution)
endif()

ExternalProject_Add(ext_rocalution
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${rocalution_SOURCE_DIR}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${ROCALUTION_INSTALL_DIR}
  -DCMAKE_INSTALL_LIBDIR=${ROCALUTION_LIBDIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_CXX_COMPILER=hipcc
  -DSUPPORT_HIP=ON
  -DSUPPORT_OMP=OFF
  -DSUPPORT_MPI=OFF
  -DBUILD_SHARED_LIBS=ON
  -DBUILD_CLIENTS_SAMPLES=ON
)

add_dependencies(lsbench ext_rocalution)
target_link_libraries(lsbench PUBLIC
  ${ROCALUTION_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rocalution_hip${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${ROCALUTION_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rocalution${CMAKE_SHARED_LIBRARY_SUFFIX})

target_include_directories(lsbench PRIVATE ${ROCALUTION_INCDIR})





