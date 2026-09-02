cmake_minimum_required(VERSION 3.15)
project(fs2d LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)   # matches vendor/fourier-soft-2d/src/CMakeLists.txt
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies of the vendored fourier-soft-2d algorithm (see that
# submodule's own src/CMakeLists.txt for PCL/OpenCV/OpenMP -- CGAL isn't
# in its find_package() calls but softDescriptorRegistration.h #includes
# CGAL headers unconditionally, so it's needed here too even though
# upstream's own build script doesn't ask for it explicitly).
find_package(PCL 1.10 REQUIRED COMPONENTS common io)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)

set(FS2D_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/fourier-soft-2d/src)

set(FS2D_VENDOR_SOURCES
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/rotate_so3_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/rotate_so3_utils.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/s2_cospmls.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/s2_legendreTransforms.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/s2_semi_fly.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/s2_semi_memo.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/so3_correlate_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/soft_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/soft_fftw_nt.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/soft_fftw_pc.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/utils_vec_cx.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/wignerTransforms_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/wrap_s2_rotate_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/wrap_soft_fftw.c
    ${FS2D_VENDOR_DIR}/soft20/src/lib1/wrap_soft_fftw_cor2.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/csecond.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/makeweights.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/makeWigner.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/s2_primitive.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/utils_so3.c
    ${FS2D_VENDOR_DIR}/soft20/src/common/wignerTransforms.c
    ${FS2D_VENDOR_DIR}/registration/src/softCorrelationClass.cpp
    ${FS2D_VENDOR_DIR}/registration/src/softDescriptorRegistration.cpp
    ${FS2D_VENDOR_DIR}/tools/src/PeakFinder.cpp
)

add_library(fs2d SHARED
    fs2d_c_api.cpp
    ${FS2D_VENDOR_SOURCES}
)

target_include_directories(fs2d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${FS2D_VENDOR_DIR}/tools/include
    ${FS2D_VENDOR_DIR}/soft20/include
    ${FS2D_VENDOR_DIR}/soft20/include/soft20
    ${FS2D_VENDOR_DIR}/registration/include
)
target_include_directories(fs2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fs2d PRIVATE
    fftw3
    OpenMP::OpenMP_CXX
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    CGAL::CGAL
    Eigen3::Eigen
)

set_target_properties(fs2d PROPERTIES
    OUTPUT_NAME "fs2d"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

