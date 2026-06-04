#
#  MicroHH
#  Copyright (c) 2011-2024 Chiel van Heerwaarden
#  Copyright (c) 2011-2024 Thijs Heus
#  Copyright (c) 2014-2024 Bart van Stratum
#
#  This file is part of MicroHH
#
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
#
include_directories(${INCLUDE_DIRS} "../include" ".")

# retrieve the git hash from the current commit
find_package(Git)
if(GIT_EXECUTABLE)
  execute_process(
    COMMAND git describe --dirty --tags
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE GITHASH
    RESULT_VARIABLE GITRETURN)
  if(NOT GITRETURN MATCHES 0)
    set(GITHASH "not available")
  endif()
else()
  set(GITHASH "not available")
endif()

message(STATUS "Git hash " ${GITHASH})

# send a precompiler statement replacing the git hash
add_definitions(-DGITHASH="${GITHASH}")

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(VTK CONFIG REQUIRED
  COMPONENTS
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    FiltersModeling
    FiltersSources
    RenderingCore
    RenderingFreeType
    RenderingOpenGL2)

add_executable(
  microhh_viz
  microhh_viz.cxx
  model_viz.cxx
  visualization_viz.cxx)

target_include_directories(
  microhh_viz
  PUBLIC
    ${INCLUDE_DIRS}
    "."
    "../include"
    "../rte-rrtmgp-cpp/include")

set(MICROHH_VIZ_LIBS
  ${VTK_LIBRARIES}
  glfw
  imgui::imgui
  OpenGL::GL)

if(USECUDA)
    target_link_libraries(microhh_viz PRIVATE microhhc curand rte_rrtmgp rte_rrtmgp_cuda rte_rrtmgp_cuda_rt rte_rrtmgp_kernels rte_rrtmgp_kernels_cuda rte_rrtmgp_kernels_cuda_rt ${LIBS} ${MICROHH_VIZ_LIBS})
else()
    target_link_libraries(microhh_viz PRIVATE microhhc rte_rrtmgp_kernels rte_rrtmgp ${LIBS} ${MICROHH_VIZ_LIBS})
endif()

vtk_module_autoinit(
  TARGETS microhh_viz
  MODULES ${VTK_LIBRARIES})
