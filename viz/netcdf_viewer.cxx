/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "netcdf_viewer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <vtkActor.h>
#include <vtkArrowSource.h>
#include <vtkCamera.h>
#include <vtkCallbackCommand.h>
#include <vtkColorTransferFunction.h>
#include <vtkCommand.h>
#include <vtkFloatArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGlyph3DMapper.h>
#include <vtkImageData.h>
#include <vtkLight.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkOpenGLState.h>
#include <vtkOutlineFilter.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRungeKutta4.h>
#include <vtkSmartPointer.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkStreamTracer.h>
#include <vtkStructuredGrid.h>
#include <vtkTextureObject.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

namespace microhh::viz
{
namespace
{
    enum class Vector_mode { Glyphs = 0, Streamlines = 1 };

    struct Camera_drag
    {
        bool active = false;
        double x = 0.;
        double y = 0.;
    };

struct Viz_state
{
    GLFWwindow* window = nullptr;

    vtkSmartPointer<vtkGenericOpenGLRenderWindow> render_window;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkStructuredGrid> grid;
    vtkSmartPointer<vtkOutlineFilter> outline;
    vtkSmartPointer<vtkPolyDataMapper> outline_mapper;
    vtkSmartPointer<vtkActor> outline_actor;
    vtkSmartPointer<vtkArrowSource> arrow;
    vtkSmartPointer<vtkPolyData> vector_data;
    vtkSmartPointer<vtkGlyph3DMapper> vector_mapper;
    vtkSmartPointer<vtkActor> vector_actor;
    vtkSmartPointer<vtkPolyData> streamline_seed_data;
    vtkSmartPointer<vtkRungeKutta4> stream_integrator;
    vtkSmartPointer<vtkStreamTracer> stream_tracer;
    vtkSmartPointer<vtkPolyDataMapper> stream_mapper;
    vtkSmartPointer<vtkActor> stream_actor;
    vtkSmartPointer<vtkImageData> volume_image;
    vtkSmartPointer<vtkSmartVolumeMapper> volume_mapper;
    vtkSmartPointer<vtkVolume> volume_actor;
    vtkSmartPointer<vtkVolumeProperty> volume_property;
    vtkSmartPointer<vtkColorTransferFunction> volume_color;
    vtkSmartPointer<vtkPiecewiseFunction> volume_opacity;
    vtkSmartPointer<vtkLight> sun_light;
    vtkSmartPointer<vtkLookupTable> vector_lut;
    vtkSmartPointer<vtkCallbackCommand> make_current_callback;
    vtkSmartPointer<vtkCallbackCommand> is_current_callback;
    vtkSmartPointer<vtkCallbackCommand> supports_opengl_callback;
    vtkSmartPointer<vtkCallbackCommand> is_direct_callback;
    vtkSmartPointer<vtkCallbackCommand> frame_callback;

    Case_settings case_settings;
    std::vector<std::string> scalar_names;
    bool show_scalar = true;
    bool show_vectors = false;
    int vector_mode = static_cast<int>(Vector_mode::Glyphs);
    int scalar_index = 0;
    int time_index = 0;
    int max_time_index = 0;
    int stride = 1;
    int vector_stride = 8;
    int streamline_stride = 10;
    bool playing = false;
    bool needs_dataset_update = true;
    bool gui_input_active = false;
    bool reset_camera = true;
    float glyph_scale = 5.f;
    float playback_rate = 8.f;
    float scattering_blending = 1.f;
    float shadow_reach = 0.5f;
    Output_grid output_grid;
    float output_top_input = 1.f;
    int vertical_cells_input = 1;
    fs::path export_directory;
    std::string export_message;
    bool export_failed = false;

    std::array<int, 3> dims = {{1, 1, 1}};
    std::array<double, 3> domain_origin = {{0., 0., 0.}};
    std::array<double, 3> domain_size = {{1., 1., 1.}};
    float scalar_min = 0.f;
    float scalar_max = 0.f;
    float vector_min = 0.f;
    float vector_max = 0.f;
    double time = 0.;
    double max_extent = 1.;

    Camera_drag orbit;
    Camera_drag pan;
    std::chrono::steady_clock::time_point last_step = std::chrono::steady_clock::now();

    bool vtk_gl_initialized = false;
};

void vtk_make_current(vtkObject*, unsigned long, void* client_data, void*)
{
    auto* state = static_cast<Viz_state*>(client_data);
    glfwMakeContextCurrent(state->window);
}

void vtk_is_current(vtkObject*, unsigned long, void* client_data, void* call_data)
{
    auto* state = static_cast<Viz_state*>(client_data);
    auto* is_current = static_cast<bool*>(call_data);
    *is_current = glfwGetCurrentContext() == state->window;
}

void vtk_supports_opengl(vtkObject*, unsigned long, void*, void* call_data)
{
    auto* supports_opengl = static_cast<int*>(call_data);
    *supports_opengl = 1;
}

void vtk_is_direct(vtkObject*, unsigned long, void*, void* call_data)
{
    auto* is_direct = static_cast<int*>(call_data);
    *is_direct = 1;
}

void vtk_frame(vtkObject*, unsigned long, void*, void*)
{
    glFlush();
}

void ensure_vtk_opengl_state(Viz_state& state)
{
    glfwMakeContextCurrent(state.window);

    if (!state.vtk_gl_initialized)
    {
        state.render_window->OpenGLInit();
        state.render_window->OpenGLInitState();
        state.vtk_gl_initialized = true;
    }

    if (!state.render_window->GetState())
        throw std::runtime_error("VTK OpenGL state is not initialized");
}

void bind_glfw_framebuffer(Viz_state& state, const int width, const int height)
{
    ensure_vtk_opengl_state(state);

    auto* gl_state = state.render_window->GetState();
    gl_state->Reset();
    gl_state->vtkglBindFramebuffer(GL_FRAMEBUFFER, 0);
    gl_state->vtkglViewport(0, 0, width, height);
}

void draw_vtk_background(Viz_state& state, const int width, const int height)
{
    auto* display_framebuffer = state.render_window->GetDisplayFramebuffer();
    if (!display_framebuffer)
        return;

    auto* texture = display_framebuffer->GetColorAttachmentAsTextureObject(0);
    if (!texture || texture->GetHandle() == 0)
        return;

    ImGui::GetBackgroundDrawList()->AddImage(
            (ImTextureID)(intptr_t)texture->GetHandle(),
            ImVec2(0.f, 0.f),
            ImVec2(static_cast<float>(width), static_cast<float>(height)),
            ImVec2(0.f, 1.f),
            ImVec2(1.f, 0.f));
}

double norm3(const double v[3])
{
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void normalize3(double v[3])
{
    const double n = norm3(v);
    if (n == 0.)
        return;

    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

void cross3(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

void zero_camera_roll(vtkCamera* camera)
{
    double pos[3];
    double focal[3];
    camera->GetPosition(pos);
    camera->GetFocalPoint(focal);

    double view_dir[3] = {
            focal[0] - pos[0],
            focal[1] - pos[1],
            focal[2] - pos[2]};
    normalize3(view_dir);

    const double world_up[3] = {0., 0., 1.};
    double right[3];
    cross3(view_dir, world_up, right);
    if (norm3(right) < 1.e-8)
    {
        camera->SetViewUp(0., 1., 0.);
        return;
    }

    normalize3(right);
    double up[3];
    cross3(right, view_dir, up);
    normalize3(up);
    camera->SetViewUp(up[0], up[1], up[2]);
}

Vector_mode current_vector_mode(Viz_state& state)
{
    state.vector_mode = std::clamp(state.vector_mode, 0, 1);
    return static_cast<Vector_mode>(state.vector_mode);
}

void update_visibility(Viz_state& state)
{
    const auto vector_mode = current_vector_mode(state);
    const bool has_velocity = state.vector_data && state.vector_data->GetNumberOfPoints() > 0;

    state.volume_actor->SetVisibility(state.show_scalar ? 1 : 0);
    state.vector_actor->SetVisibility(
            state.show_vectors && vector_mode == Vector_mode::Glyphs && has_velocity ? 1 : 0);
    state.stream_actor->SetVisibility(
            state.show_vectors && vector_mode == Vector_mode::Streamlines && has_velocity ? 1 : 0);
}

void update_volume_transfer(Viz_state& state)
{
    const double max_value = std::max(1.e-12, static_cast<double>(state.scalar_max));

    state.volume_color->RemoveAllPoints();
    state.volume_color->AddRGBPoint(0., 0.86, 0.89, 0.92);
    state.volume_color->AddRGBPoint(max_value, 1.00, 1.00, 0.98);

    state.volume_opacity->RemoveAllPoints();
    for (int i=0; i<=256; ++i)
    {
        const double extinction = max_value * static_cast<double>(i) / 256.;
        state.volume_opacity->AddPoint(extinction, -std::expm1(-extinction));
    }
    state.volume_property->SetScalarOpacityUnitDistance(1.);
    state.volume_property->Modified();
}

void update_volume_lighting(Viz_state& state)
{
    state.volume_mapper->SetVolumetricScatteringBlending(state.scattering_blending);
    state.volume_mapper->SetGlobalIlluminationReach(state.shadow_reach);
    state.volume_mapper->Modified();
}

void update_streamline_settings(Viz_state& state)
{
    const double length = std::max(1.e-6, state.max_extent);
    state.stream_tracer->SetMaximumPropagation(2.5*length);
    state.stream_tracer->SetIntegrationStepUnit(vtkStreamTracer::LENGTH_UNIT);
    state.stream_tracer->SetInitialIntegrationStep(0.01*length);
    state.stream_tracer->SetMinimumIntegrationStep(0.001*length);
    state.stream_tracer->SetMaximumIntegrationStep(0.05*length);
    state.stream_tracer->SetMaximumNumberOfSteps(2000);
    state.stream_tracer->SetTerminalSpeed(1.e-12);
    state.stream_tracer->Modified();
}

void update_volume_image(Viz_state& state, const Snapshot& snapshot, vtkFloatArray* scalars)
{
    state.volume_image->SetDimensions(snapshot.nx, snapshot.ny, snapshot.nz);
    state.volume_image->SetOrigin(
            snapshot.domain_origin[0] + 0.5*snapshot.cell_size[0],
            snapshot.domain_origin[1] + 0.5*snapshot.cell_size[1],
            snapshot.domain_origin[2] + 0.5*snapshot.cell_size[2]);
    state.volume_image->SetSpacing(
            snapshot.cell_size[0], snapshot.cell_size[1], snapshot.cell_size[2]);
    state.volume_image->GetPointData()->SetScalars(scalars);

    state.volume_image->Modified();
}

void update_velocity_polydata(Viz_state& state, const Snapshot& snapshot)
{
    vtkNew<vtkPoints> vector_points;
    vector_points->SetDataTypeToFloat();

    vtkNew<vtkFloatArray> vector_vectors;
    vtkNew<vtkFloatArray> vector_magnitude;
    vector_vectors->SetName("velocity");
    vector_vectors->SetNumberOfComponents(3);
    vector_magnitude->SetName("velocity_magnitude");
    vector_magnitude->SetNumberOfComponents(1);

    if (!snapshot.vectors.empty())
    {
        const int stride = std::max(1, state.vector_stride);
        for (int k=0; k<snapshot.nz; k += stride)
            for (int j=0; j<snapshot.ny; j += stride)
                for (int i=0; i<snapshot.nx; i += stride)
                {
                    const auto id = point_id(snapshot.nx, snapshot.ny, i, j, k);
                    const std::size_t offset = 3*id;
                    const float u = snapshot.vectors[offset];
                    const float v = snapshot.vectors[offset+1];
                    const float w = snapshot.vectors[offset+2];
                    const float mag = std::sqrt(u*u + v*v + w*w);

                    vector_points->InsertNextPoint(
                            snapshot.points[offset],
                            snapshot.points[offset+1],
                            snapshot.points[offset+2]);
                    vector_vectors->InsertNextTuple3(u, v, w);
                    vector_magnitude->InsertNextValue(mag);
                }
    }

    state.vector_data->SetPoints(vector_points);
    state.vector_data->GetPointData()->SetVectors(vector_vectors);
    state.vector_data->GetPointData()->AddArray(vector_magnitude);
    state.vector_data->Modified();
}

void update_streamline_seeds(Viz_state& state, const Snapshot& snapshot)
{
    vtkNew<vtkPoints> seed_points;
    seed_points->SetDataTypeToFloat();

    if (!snapshot.vectors.empty())
    {
        const int stride = std::max(1, state.streamline_stride);
        for (int k=0; k<snapshot.nz; k += stride)
            for (int j=0; j<snapshot.ny; j += stride)
                for (int i=0; i<snapshot.nx; i += stride)
                {
                    const auto id = point_id(snapshot.nx, snapshot.ny, i, j, k);
                    const std::size_t offset = 3*id;
                    seed_points->InsertNextPoint(
                            snapshot.points[offset],
                            snapshot.points[offset+1],
                            snapshot.points[offset+2]);
                }
    }

    state.streamline_seed_data->SetPoints(seed_points);
    state.streamline_seed_data->Modified();
}

void update_dataset(Viz_state& state, const Dataset& dataset)
{
    state.scalar_index = std::clamp(
            state.scalar_index, 0, static_cast<int>(state.scalar_names.size())-1);
    const auto& scalar_name = state.scalar_names[static_cast<std::size_t>(state.scalar_index)];
    state.max_time_index = std::max(0, dataset.time_count(scalar_name) - 1);
    state.time_index = std::clamp(state.time_index, 0, state.max_time_index);

    const bool include_velocity = state.show_vectors;
    const auto snapshot = dataset.snapshot(
            scalar_name, state.time_index, state.stride, include_velocity, state.output_grid);
    state.dims = {{snapshot.nx, snapshot.ny, snapshot.nz}};
    state.domain_origin = snapshot.domain_origin;
    state.domain_size = snapshot.domain_size;
    state.scalar_min = snapshot.scalar_min;
    state.scalar_max = snapshot.scalar_max;
    state.vector_min = snapshot.vector_min;
    state.vector_max = snapshot.vector_max;
    state.time = snapshot.time;

    vtkNew<vtkPoints> points;
    points->SetDataTypeToFloat();
    points->SetNumberOfPoints(static_cast<vtkIdType>(snapshot.scalars.size()));

    for (vtkIdType id=0; id<static_cast<vtkIdType>(snapshot.scalars.size()); ++id)
    {
        const std::size_t offset = static_cast<std::size_t>(3*id);
        points->SetPoint(
                id,
                snapshot.points[offset],
                snapshot.points[offset+1],
                snapshot.points[offset+2]);
    }

    vtkNew<vtkFloatArray> scalars;
    scalars->SetName(scalar_name.c_str());
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(static_cast<vtkIdType>(snapshot.scalars.size()));
    for (vtkIdType id=0; id<static_cast<vtkIdType>(snapshot.scalars.size()); ++id)
        scalars->SetValue(id, snapshot.scalars[static_cast<std::size_t>(id)]);

    vtkNew<vtkFloatArray> vectors;
    vtkNew<vtkFloatArray> vector_magnitude;
    vectors->SetName("velocity");
    vectors->SetNumberOfComponents(3);
    vector_magnitude->SetName("velocity_magnitude");
    vector_magnitude->SetNumberOfComponents(1);

    if (!snapshot.vectors.empty())
    {
        const vtkIdType ntuples = static_cast<vtkIdType>(snapshot.vectors.size()/3);
        vectors->SetNumberOfTuples(ntuples);
        vector_magnitude->SetNumberOfTuples(ntuples);

        for (vtkIdType id=0; id<ntuples; ++id)
        {
            const std::size_t offset = static_cast<std::size_t>(3*id);
            const float u = snapshot.vectors[offset];
            const float v = snapshot.vectors[offset+1];
            const float w = snapshot.vectors[offset+2];
            const float mag = std::sqrt(u*u + v*v + w*w);

            vectors->SetTuple3(id, u, v, w);
            vector_magnitude->SetValue(id, mag);
        }
    }

    state.grid->SetDimensions(snapshot.nx, snapshot.ny, snapshot.nz);
    state.grid->SetPoints(points);
    state.grid->GetPointData()->SetScalars(scalars);
    if (!snapshot.vectors.empty())
    {
        state.grid->GetPointData()->SetVectors(vectors);
        state.grid->GetPointData()->AddArray(vector_magnitude);
    }
    else
        state.grid->GetPointData()->SetVectors(nullptr);

    state.grid->Modified();
    update_volume_image(state, snapshot, scalars);
    update_velocity_polydata(state, snapshot);
    update_streamline_seeds(state, snapshot);

    const float vector_range_min = state.vector_min == state.vector_max
        ? state.vector_min - 1.f : state.vector_min;
    const float vector_range_max = state.vector_min == state.vector_max
        ? state.vector_max + 1.f : state.vector_max;
    state.vector_lut->SetTableRange(vector_range_min, vector_range_max);
    state.vector_lut->Build();
    state.vector_mapper->SetScalarRange(vector_range_min, vector_range_max);
    state.stream_mapper->SetScalarRange(vector_range_min, vector_range_max);

    double bounds[6] = {0., 1., 0., 1., 0., 1.};
    state.grid->GetBounds(bounds);
    state.max_extent = std::max({
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
            1.});
    state.vector_mapper->SetScaleFactor(state.glyph_scale * state.max_extent * 0.04);
    update_volume_transfer(state);
    update_streamline_settings(state);
    update_visibility(state);

    state.needs_dataset_update = false;

    if (state.reset_camera)
    {
        state.renderer->ResetCamera();
        zero_camera_roll(state.renderer->GetActiveCamera());
        state.renderer->ResetCameraClippingRange();
        state.reset_camera = false;
    }
}

void apply_mouse_camera(Viz_state& state)
{
    if (ImGui::GetIO().WantCaptureMouse)
    {
        state.orbit.active = false;
        state.pan.active = false;
        return;
    }

    double x = 0.;
    double y = 0.;
    glfwGetCursorPos(state.window, &x, &y);

    auto* camera = state.renderer->GetActiveCamera();

    const bool left_down = glfwGetMouseButton(state.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (left_down)
    {
        if (state.orbit.active)
        {
            const double dx = x - state.orbit.x;
            const double dy = y - state.orbit.y;
            camera->Azimuth(-0.35*dx);
            camera->Elevation(0.35*dy);
            zero_camera_roll(camera);
            state.renderer->ResetCameraClippingRange();
        }

        state.orbit = {true, x, y};
    }
    else
        state.orbit.active = false;

    const bool right_down = glfwGetMouseButton(state.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (right_down)
    {
        if (state.pan.active)
        {
            double pos[3];
            double focal[3];
            camera->GetPosition(pos);
            camera->GetFocalPoint(focal);

            double view_dir[3] = {
                    focal[0] - pos[0],
                    focal[1] - pos[1],
                    focal[2] - pos[2]};
            const double distance = std::max(1.e-6, norm3(view_dir));
            normalize3(view_dir);

            const double world_up[3] = {0., 0., 1.};
            double right[3];
            cross3(view_dir, world_up, right);
            if (norm3(right) < 1.e-8)
                right[0] = 1.;
            normalize3(right);

            double up[3];
            cross3(right, view_dir, up);
            normalize3(up);

            const double dx = x - state.pan.x;
            const double dy = y - state.pan.y;
            const double scale = 0.0015 * distance;
            const double shift[3] = {
                    (-dx*right[0] + dy*up[0]) * scale,
                    (-dx*right[1] + dy*up[1]) * scale,
                    (-dx*right[2] + dy*up[2]) * scale};

            camera->SetPosition(pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]);
            camera->SetFocalPoint(focal[0] + shift[0], focal[1] + shift[1], focal[2] + shift[2]);
            zero_camera_roll(camera);
            state.renderer->ResetCameraClippingRange();
        }

        state.pan = {true, x, y};
    }
    else
        state.pan.active = false;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

    if (action != GLFW_PRESS)
        return;

    auto* state = static_cast<Viz_state*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else if (key == GLFW_KEY_SPACE)
        state->playing = !state->playing;
    else if (key == GLFW_KEY_N || key == GLFW_KEY_RIGHT)
    {
        state->time_index = std::min(state->time_index + 1, state->max_time_index);
        state->needs_dataset_update = true;
    }
    else if (key == GLFW_KEY_LEFT)
    {
        state->time_index = std::max(state->time_index - 1, 0);
        state->needs_dataset_update = true;
    }
    else if (key == GLFW_KEY_R)
    {
        state->reset_camera = true;
        state->needs_dataset_update = true;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

    auto* state = static_cast<Viz_state*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse)
    {
        state->orbit.active = false;
        state->pan.active = false;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    auto* state = static_cast<Viz_state*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    auto* camera = state->renderer->GetActiveCamera();
    camera->Dolly(yoffset > 0. ? 1.12 : 0.88);
    zero_camera_roll(camera);
    state->renderer->ResetCameraClippingRange();
}

void char_callback(GLFWwindow* window, unsigned int c)
{
    ImGui_ImplGlfw_CharCallback(window, c);
}

void create_pipeline(Viz_state& state)
{
    state.render_window = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    state.renderer = vtkSmartPointer<vtkRenderer>::New();
    state.grid = vtkSmartPointer<vtkStructuredGrid>::New();
    state.outline = vtkSmartPointer<vtkOutlineFilter>::New();
    state.outline_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    state.outline_actor = vtkSmartPointer<vtkActor>::New();
    state.arrow = vtkSmartPointer<vtkArrowSource>::New();
    state.vector_data = vtkSmartPointer<vtkPolyData>::New();
    state.vector_mapper = vtkSmartPointer<vtkGlyph3DMapper>::New();
    state.vector_actor = vtkSmartPointer<vtkActor>::New();
    state.streamline_seed_data = vtkSmartPointer<vtkPolyData>::New();
    state.stream_integrator = vtkSmartPointer<vtkRungeKutta4>::New();
    state.stream_tracer = vtkSmartPointer<vtkStreamTracer>::New();
    state.stream_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    state.stream_actor = vtkSmartPointer<vtkActor>::New();
    state.volume_image = vtkSmartPointer<vtkImageData>::New();
    state.volume_mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
    state.volume_actor = vtkSmartPointer<vtkVolume>::New();
    state.volume_property = vtkSmartPointer<vtkVolumeProperty>::New();
    state.volume_color = vtkSmartPointer<vtkColorTransferFunction>::New();
    state.volume_opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
    state.sun_light = vtkSmartPointer<vtkLight>::New();
    state.vector_lut = vtkSmartPointer<vtkLookupTable>::New();

    state.make_current_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    state.make_current_callback->SetClientData(&state);
    state.make_current_callback->SetCallback(vtk_make_current);

    state.is_current_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    state.is_current_callback->SetClientData(&state);
    state.is_current_callback->SetCallback(vtk_is_current);

    state.supports_opengl_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    state.supports_opengl_callback->SetCallback(vtk_supports_opengl);

    state.is_direct_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    state.is_direct_callback->SetCallback(vtk_is_direct);

    state.frame_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    state.frame_callback->SetCallback(vtk_frame);

    state.render_window->AddObserver(vtkCommand::WindowMakeCurrentEvent, state.make_current_callback);
    state.render_window->AddObserver(vtkCommand::WindowIsCurrentEvent, state.is_current_callback);
    state.render_window->AddObserver(vtkCommand::WindowSupportsOpenGLEvent, state.supports_opengl_callback);
    state.render_window->AddObserver(vtkCommand::WindowIsDirectEvent, state.is_direct_callback);
    state.render_window->AddObserver(vtkCommand::WindowFrameEvent, state.frame_callback);
    state.render_window->SetOwnContext(false);
    state.render_window->SetOffScreenRendering(true);
    state.render_window->SetFrameBlitModeToNoBlit();
    state.render_window->SetReadyForRendering(true);
    state.render_window->SetSwapBuffers(true);
    state.render_window->AddRenderer(state.renderer);
    state.renderer->SetBackground(0.08, 0.09, 0.1);
    state.renderer->AutomaticLightCreationOff();
    state.sun_light->SetLightTypeToSceneLight();
    state.sun_light->SetPositional(false);
    state.sun_light->SetPosition(-1., -1., 2.);
    state.sun_light->SetFocalPoint(0., 0., 0.);
    state.sun_light->SetColor(1., 0.98, 0.92);
    state.sun_light->SetIntensity(1.2);
    state.renderer->AddLight(state.sun_light);

    state.vector_lut->SetHueRange(0.32, 0.0);
    state.vector_lut->SetSaturationRange(0.8, 0.9);
    state.vector_lut->SetValueRange(0.9, 0.95);
    state.vector_lut->Build();

    state.outline->SetInputData(state.grid);
    state.outline_mapper->SetInputConnection(state.outline->GetOutputPort());
    state.outline_actor->SetMapper(state.outline_mapper);
    state.outline_actor->GetProperty()->SetColor(0.88, 0.88, 0.82);
    state.outline_actor->GetProperty()->SetLineWidth(1.5);

    state.vector_mapper->SetInputData(state.vector_data);
    state.vector_mapper->SetSourceConnection(state.arrow->GetOutputPort());
    state.vector_mapper->SetLookupTable(state.vector_lut);
    state.vector_mapper->SetOrientationArray("velocity");
    state.vector_mapper->SetOrientationModeToDirection();
    state.vector_mapper->SetScaleArray("velocity_magnitude");
    state.vector_mapper->SetScaleModeToScaleByMagnitude();
    state.vector_mapper->SetScalarModeToUsePointFieldData();
    state.vector_mapper->SelectColorArray("velocity_magnitude");
    state.vector_mapper->OrientOn();
    state.vector_mapper->ScalingOn();
    state.vector_actor->SetMapper(state.vector_mapper);

    state.stream_tracer->SetInputData(state.grid);
    state.stream_tracer->SetSourceData(state.streamline_seed_data);
    state.stream_tracer->SetIntegrator(state.stream_integrator);
    state.stream_tracer->SetIntegrationDirectionToBoth();
    state.stream_mapper->SetInputConnection(state.stream_tracer->GetOutputPort());
    state.stream_mapper->SetLookupTable(state.vector_lut);
    state.stream_mapper->SetScalarModeToUsePointFieldData();
    state.stream_mapper->SelectColorArray("velocity_magnitude");
    state.stream_mapper->ScalarVisibilityOn();
    state.stream_actor->SetMapper(state.stream_mapper);
    state.stream_actor->GetProperty()->SetLineWidth(1.3);

    state.volume_mapper->SetInputData(state.volume_image);
    state.volume_mapper->SetBlendModeToComposite();
    state.volume_mapper->SetRequestedRenderModeToGPU();
    state.volume_mapper->SetUseJittering(true);
    state.volume_property->SetColor(state.volume_color);
    state.volume_property->SetScalarOpacity(state.volume_opacity);
    state.volume_property->SetInterpolationTypeToLinear();
    state.volume_property->ShadeOn();
    state.volume_property->SetAmbient(0.1);
    state.volume_property->SetDiffuse(0.9);
    state.volume_property->SetSpecular(0.0);
    update_volume_lighting(state);
    state.volume_actor->SetMapper(state.volume_mapper);
    state.volume_actor->SetProperty(state.volume_property);

    state.renderer->AddVolume(state.volume_actor);
    state.renderer->AddActor(state.outline_actor);
    state.renderer->AddActor(state.vector_actor);
    state.renderer->AddActor(state.stream_actor);
}

void render_gui(Viz_state& state, const Dataset& dataset)
{
    ImGui::SetNextWindowPos(ImVec2(14.f, 14.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.f, 0.f), ImGuiCond_FirstUseEver);
    ImGui::Begin("MicroHH NetCDF Viz");

    ImGui::Text("time %.6g  frame %d/%d", state.time, state.time_index, state.max_time_index);
    if (!state.case_settings.ini_path.empty())
    {
        ImGui::Text("ini %s", state.case_settings.ini_path.filename().string().c_str());
        if (state.case_settings.has_domain_size)
            ImGui::Text(
                    "domain %.6g x %.6g x %.6g m",
                    state.case_settings.domain_size[0],
                    state.case_settings.domain_size[1],
                    state.case_settings.domain_size[2]);
        if (state.case_settings.has_grid_size)
            ImGui::Text(
                    "grid %.0f x %.0f x %.0f",
                    state.case_settings.grid_size[0],
                    state.case_settings.grid_size[1],
                    state.case_settings.grid_size[2]);
    }

    if (ImGui::Button(state.playing ? "Pause" : "Play"))
        state.playing = !state.playing;
    ImGui::SameLine();
    if (ImGui::Button("Prev"))
    {
        state.time_index = std::max(state.time_index - 1, 0);
        state.needs_dataset_update = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Next"))
    {
        state.time_index = std::min(state.time_index + 1, state.max_time_index);
        state.needs_dataset_update = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        state.reset_camera = true;
        state.needs_dataset_update = true;
    }

    ImGui::SliderInt("Frame", &state.time_index, 0, state.max_time_index);
    if (ImGui::IsItemDeactivatedAfterEdit())
        state.needs_dataset_update = true;
    ImGui::SliderFloat("Playback", &state.playback_rate, 1.f, 60.f, "%.0f fps");

    if (ImGui::Checkbox("Show Scalar", &state.show_scalar))
        update_visibility(state);
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Vector", &state.show_vectors))
    {
        state.needs_dataset_update = true;
        update_visibility(state);
    }

    const char* vector_mode_labels[] = {"Glyphs", "Streamlines"};
    if (state.show_vectors && ImGui::Combo("Vector mode", &state.vector_mode, vector_mode_labels, 2))
        update_visibility(state);

    const auto vector_mode = current_vector_mode(state);
    const bool glyph_method = state.show_vectors && vector_mode == Vector_mode::Glyphs;
    const bool streamline_method = state.show_vectors && vector_mode == Vector_mode::Streamlines;

    std::vector<const char*> scalar_labels;
    scalar_labels.reserve(state.scalar_names.size());
    for (const auto& name : state.scalar_names)
        scalar_labels.push_back(name.c_str());

    if (ImGui::Combo(
                "Scalar",
                &state.scalar_index,
                scalar_labels.data(),
                static_cast<int>(scalar_labels.size())))
    {
        const auto& scalar_name = state.scalar_names[static_cast<std::size_t>(state.scalar_index)];
        state.max_time_index = std::max(0, dataset.time_count(scalar_name) - 1);
        state.time_index = std::clamp(state.time_index, 0, state.max_time_index);
        state.needs_dataset_update = true;
    }

    ImGui::SliderInt("Grid stride", &state.stride, 1, 16);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        state.needs_dataset_update = true;
        state.reset_camera = true;
    }

    bool grid_edit_finished = false;
    ImGui::InputFloat("Output top (m)", &state.output_top_input, 10.f, 100.f, "%.6g");
    grid_edit_finished |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::InputInt("Vertical cells", &state.vertical_cells_input);
    grid_edit_finished |= ImGui::IsItemDeactivatedAfterEdit();
    if (grid_edit_finished)
    {
        state.output_top_input = std::max(1.e-6f, state.output_top_input);
        state.vertical_cells_input = std::max(1, state.vertical_cells_input);
        state.output_grid.top = state.output_top_input;
        state.output_grid.vertical_cells = state.vertical_cells_input;
        state.needs_dataset_update = true;
        state.reset_camera = true;
    }

    bool lighting_edit_finished = false;
    ImGui::SliderFloat(
            "Volume scattering", &state.scattering_blending, 0.f, 2.f, "%.2f");
    lighting_edit_finished |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SliderFloat(
            "Shadow reach", &state.shadow_reach, 0.f, 1.f, "%.2f");
    lighting_edit_finished |= ImGui::IsItemDeactivatedAfterEdit();
    if (lighting_edit_finished)
        update_volume_lighting(state);

    ImGui::Text(
            "origin (%.6g, %.6g, %.6g) m",
            state.domain_origin[0], state.domain_origin[1], state.domain_origin[2]);
    ImGui::Text(
            "size (%.6g, %.6g, %.6g) m",
            state.domain_size[0], state.domain_size[1], state.domain_size[2]);

    if (glyph_method)
    {
        ImGui::SliderInt("Vector stride", &state.vector_stride, 1, 64);
        if (ImGui::IsItemDeactivatedAfterEdit())
            state.needs_dataset_update = true;
    }

    if (glyph_method)
    {
        ImGui::SliderFloat("Vector scale", &state.glyph_scale, 0.05f, 100.f);
        if (ImGui::IsItemDeactivatedAfterEdit())
            state.vector_mapper->SetScaleFactor(state.glyph_scale * state.max_extent * 0.04);
    }

    if (streamline_method)
    {
        ImGui::SliderInt("Seed stride", &state.streamline_stride, 1, 64);
        if (ImGui::IsItemDeactivatedAfterEdit())
            state.needs_dataset_update = true;
    }

    if (state.show_scalar)
        ImGui::Text("extinction %.6g .. %.6g m^-1", state.scalar_min, state.scalar_max);

    if (state.show_vectors)
        ImGui::Text("speed %.6g .. %.6g", state.vector_min, state.vector_max);

    if (state.show_vectors && (!dataset.has_velocity() || !state.vector_data || state.vector_data->GetNumberOfPoints() == 0))
        ImGui::Text("velocity unavailable");

    ImGui::Separator();
    if (!state.export_directory.empty())
        ImGui::TextWrapped("VDB dir %s", state.export_directory.string().c_str());

    const bool can_export_vdb = dataset.has_cloud_velocity_fields();
    if (!can_export_vdb)
        ImGui::BeginDisabled();

    if (ImGui::Button("Export cloud + velocity VDB"))
    {
        try
        {
            const auto summary = dataset.export_cloud_velocity_vdb_sequence(
                    state.export_directory, state.output_grid);
            std::ostringstream message;
            message << "exported " << summary.frames << " frames of ql, qi, and masked u/v/w";
            if (summary.resampled)
                message << " (resampled)";
            message << " to " << state.export_directory.string();
            state.export_message = message.str();
            state.export_failed = false;
        }
        catch (const std::exception& e)
        {
            state.export_message = e.what();
            state.export_failed = true;
        }
    }

    if (!can_export_vdb)
        ImGui::EndDisabled();

    if (!state.export_message.empty())
    {
        if (state.export_failed)
            ImGui::TextColored(ImVec4(1.f, 0.34f, 0.24f, 1.f), "%s", state.export_message.c_str());
        else
            ImGui::TextWrapped("%s", state.export_message.c_str());
    }

    state.gui_input_active = ImGui::IsAnyItemActive();
    ImGui::End();
}
}

void run_visualizer(
        const Dataset& dataset,
        const Case_settings& case_settings,
        const fs::path& source_directory,
        const Output_grid& output_grid)
{
    Viz_state state;
    state.scalar_names = dataset.scalar_names();
    if (const auto it = std::find(state.scalar_names.begin(), state.scalar_names.end(), "ql+qi");
            it != state.scalar_names.end())
        state.scalar_index = static_cast<int>(std::distance(state.scalar_names.begin(), it));
    state.case_settings = case_settings;
    state.output_grid = output_grid;
    state.output_top_input = static_cast<float>(output_grid.top);
    state.vertical_cells_input = output_grid.vertical_cells;
    state.export_directory = (source_directory.empty() ? fs::current_path() : source_directory) / "vdb";

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    state.window = glfwCreateWindow(1280, 800, "MicroHH NetCDF Viz", nullptr, nullptr);
    if (!state.window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(state.window);
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(state.window, false);
    ImGui_ImplOpenGL3_Init("#version 330");
    glfwSetWindowUserPointer(state.window, &state);
    glfwSetKeyCallback(state.window, key_callback);
    glfwSetMouseButtonCallback(state.window, mouse_button_callback);
    glfwSetScrollCallback(state.window, scroll_callback);
    glfwSetCharCallback(state.window, char_callback);

    create_pipeline(state);
    update_dataset(state, dataset);

    while (!glfwWindowShouldClose(state.window))
    {
        glfwPollEvents();

        if (state.playing)
        {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - state.last_step).count();
            const double frame_time = 1.0 / std::max(1.f, state.playback_rate);
            if (elapsed >= frame_time)
            {
                state.time_index = state.time_index >= state.max_time_index ? 0 : state.time_index + 1;
                state.needs_dataset_update = true;
                state.last_step = now;
            }
        }
        else
            state.last_step = std::chrono::steady_clock::now();

        if (state.needs_dataset_update && !state.gui_input_active)
            update_dataset(state, dataset);

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(state.window, &width, &height);
        if (width <= 0 || height <= 0)
            continue;

        glfwMakeContextCurrent(state.window);
        state.render_window->SetSize(width, height);
        bind_glfw_framebuffer(state, width, height);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        render_gui(state, dataset);
        apply_mouse_camera(state);

        glfwMakeContextCurrent(state.window);
        bind_glfw_framebuffer(state, width, height);
        state.render_window->Render();

        glfwMakeContextCurrent(state.window);
        bind_glfw_framebuffer(state, width, height);
        glClearColor(0.08f, 0.09f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_vtk_background(state, width, height);

        ImGui::Render();
        bind_glfw_framebuffer(state, width, height);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(state.window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(state.window);
    glfwTerminate();
}
}
