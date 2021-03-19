set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)


add_library(LinesMeasurerLibrary detect-lines.cpp known-good.cpp detect-lines.h known-good.h tswdft2d.h nanoflann.hpp)
target_link_libraries(LinesMeasurerLibrary
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    )

target_include_directories(LinesMeasurerLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
