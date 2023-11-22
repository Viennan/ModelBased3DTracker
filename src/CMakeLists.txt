add_library(mb3t SHARED hist_modality.cpp hist_model.cpp sparse_viewpoint_model.cpp optimizer.cpp)
target_include_directories(mb3t PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mb3t PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)