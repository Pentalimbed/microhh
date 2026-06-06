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
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(VTK CONFIG REQUIRED
  COMPONENTS
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeneral
    FiltersFlowPaths
    FiltersGeometry
    FiltersModeling
    FiltersSources
    RenderingCore
    RenderingFreeType
    RenderingOpenGL2
    RenderingVolume
    RenderingVolumeOpenGL2)

add_executable(
  microhh_nc_viz
  netcdf_visualizer.cxx)

target_include_directories(
  microhh_nc_viz
  PRIVATE
    ${NETCDF_INCLUDE_DIR}
    ${NETCDF_INCLUDE_DIRS})

target_link_libraries(
  microhh_nc_viz
  PRIVATE
    ${VTK_LIBRARIES}
    glfw
    imgui::imgui
    OpenVDB::openvdb
    OpenGL::GL
    ${NETCDF_LIB_C})

vtk_module_autoinit(
  TARGETS microhh_nc_viz
  MODULES ${VTK_LIBRARIES})
