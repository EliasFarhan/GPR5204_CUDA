cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(GPR5204_CUDA LANGUAGES CXX CUDA)



include_directories(include/)

file(GLOB_RECURSE GPR5204_SRC include/*.h src/*.cpp src/*.cu)

add_library(COMMON STATIC ${GPR5204_SRC})

set(main_project_dir "${CMAKE_SOURCE_DIR}/main/")
file(GLOB main_projects "${main_project_dir}/*.cpp")
foreach(main_project_path ${main_projects})
    # I used a simple string replace, to cut off .cpp.
    get_filename_component(main_project_name ${main_project_path} NAME )

    message("Main project path: ${main_project_path} and name: ${main_project_name}")

    add_executable(${main_project_name} ${main_project_path})
    target_link_libraries(${main_project_name} COMMON)

endforeach()