cmake_minimum_required(VERSION 3.16)
project(geolab LANGUAGES C CXX CUDA)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Release")
# Positional independent code 生成与位置无关的代码，方便别的代码引用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

################################################################################
# configure dependencies
################################################################################
include_directories("dependencies")
# Eigen
add_definitions("-DEIGEN_GPUCC")
include_directories("dependencies/eigen")
# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

add_subdirectory("dependencies/glfw")
add_subdirectory("dependencies/tinyply")

set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

mark_as_advanced(
	GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
	GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
	GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
	BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
)

include_directories("dependencies/glfw/include")
include_directories("dependencies/imgui")
include_directories("dependencies/imguizmo")
include_directories("dependencies/pybind11/include")
include_directories("dependencies/tinyply/source")
# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
set(GL_LIBRARIES GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)

# OPTIX
option(BUILD_WITH_OPTIX "if use OptiX to enable hardware ray tracing?" ON)
if(${BUILD_WITH_OPTIX})

	set(CMAKE_MODULE_PATH
	"/usr/local/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/SDK/CMake"
	${CMAKE_MODULE_PATH}
	)
	set(OptiX_INSTALL_DIR "/usr/local/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64/" CACHE PATH "Path to OptiX installed location.")
	find_package(OptiX)

	if((OptiX_FOUND OR OptiX_INCLUDE) AND BUILD_WITH_OPTIX)
		set(GEO_OPTIX ON)
		include_directories("${OptiX_INCLUDE}")
		list(APPEND GEO_DEFINITIONS -DGEO_OPTIX)
		message("Enable OptiX.")
	else()
		set(GEO_OPTIX OFF)
		if(BUILD_WITH_OPTIX)
			message(WARNING "OptiX was not found.")
		endif()
	endif()
endif()

set(GL_SOURCES
	${GL_SOURCES}
	dependencies/imguizmo/ImGuizmo.cpp
	dependencies/imgui/imgui.cpp
	dependencies/imgui/backends/imgui_impl_glfw.cpp
	dependencies/imgui/backends/imgui_impl_opengl3.cpp
	dependencies/imgui/imgui_draw.cpp
	dependencies/imgui/imgui_tables.cpp
	dependencies/imgui/imgui_widgets.cpp
)

set(SOURCES
	${GL_SOURCES}
	src/trianglebvh.cu
	src/renderer.cu
	src/voxel.cu
	src/ray_tracer_utils.cu
)

if (GEO_OPTIX)
	add_library(optix_program OBJECT
		src/optix/pathescape.cu
		src/optix/raystab.cu
		src/optix/ray_trace.cu
	)
	set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON)
	set_target_properties(optix_program PROPERTIES CUDA_ARCHITECTURES OFF)
	set_target_properties(optix_program PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/ptx)
	target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")
	add_definitions("-DGEO_OPTIX")
endif()

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

add_library(geo STATIC ${SOURCES})
set_target_properties(geo PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(geo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(geo PUBLIC ${GL_LIBRARIES} glfw)
target_compile_options(geo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

include_directories(/usr/local/include /usr/include ./include /usr/local/cuda-11.1/targets/x86_64-linux/include)

find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
	message("Python Found")
	add_subdirectory("dependencies/pybind11")
endif()

if (Python_FOUND)
	add_library(pygeo SHARED src/python-api.cu)
	set_target_properties(pygeo PROPERTIES CXX_VISIBILITY_PRESET "hidden")
	set_target_properties(pygeo PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
	target_link_libraries(pygeo PUBLIC geo ${PYTHON_LIBRARIES} pybind11::module)
	target_compile_options(pygeo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
	pybind11_extension(pygeo)
endif()