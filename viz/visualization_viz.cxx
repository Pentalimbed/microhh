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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
#include <vtkDataSetMapper.h>
#include <vtkFloatArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGlyph3DMapper.h>
#include <vtkImageData.h>
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
#include <vtkStructuredGridGeometryFilter.h>
#include <vtkTextureObject.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

#include "model_viz.h"
#include "visualization_viz.h"

namespace
{
    enum class Slice_axis { X = 0, Y = 1, Z = 2 };

    enum class Visualization_method
    {
        Scalar_slice = 0,
        Scalar_volume = 1,
        Velocity_glyphs = 2,
        Velocity_streamlines = 3
    };

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
        vtkSmartPointer<vtkStructuredGridGeometryFilter> slice;
        vtkSmartPointer<vtkDataSetMapper> slice_mapper;
        vtkSmartPointer<vtkActor> slice_actor;
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
        vtkSmartPointer<vtkImageData> scalar_image;
        vtkSmartPointer<vtkSmartVolumeMapper> volume_mapper;
        vtkSmartPointer<vtkVolume> volume_actor;
        vtkSmartPointer<vtkVolumeProperty> volume_property;
        vtkSmartPointer<vtkColorTransferFunction> volume_color;
        vtkSmartPointer<vtkPiecewiseFunction> volume_opacity;
        vtkSmartPointer<vtkLookupTable> scalar_lut;
        vtkSmartPointer<vtkLookupTable> vector_lut;
        vtkSmartPointer<vtkCallbackCommand> make_current_callback;
        vtkSmartPointer<vtkCallbackCommand> is_current_callback;
        vtkSmartPointer<vtkCallbackCommand> supports_opengl_callback;
        vtkSmartPointer<vtkCallbackCommand> is_direct_callback;
        vtkSmartPointer<vtkCallbackCommand> frame_callback;

        std::vector<std::string> scalar_names;
        int method = static_cast<int>(Visualization_method::Scalar_slice);
        int scalar_index = 0;
        int stride = 1;
        int vector_stride = 8;
        int streamline_stride = 10;
        int slice_axis = static_cast<int>(Slice_axis::Z);
        int slice_index = 0;
        bool running = false;
        bool step_requested = false;
        bool restart_requested = false;
        bool needs_dataset_update = true;
        bool reset_camera = true;
        float glyph_scale = 5.f;
        float volume_opacity_scale = 1.f;

        std::array<int, 3> dims = {{1, 1, 1}};
        float scalar_min = 0.f;
        float scalar_max = 0.f;
        float vector_min = 0.f;
        float vector_max = 0.f;
        double time = 0.;
        double dt = 0.;
        double max_extent = 1.;
        int iteration = 0;

        Camera_drag orbit;
        Camera_drag pan;

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

    void bind_glfw_framebuffer(Viz_state& state, int width, int height)
    {
        ensure_vtk_opengl_state(state);

        auto* gl_state = state.render_window->GetState();
        gl_state->Reset();
        gl_state->vtkglBindFramebuffer(GL_FRAMEBUFFER, 0);
        gl_state->vtkglViewport(0, 0, width, height);
    }

    void draw_vtk_background(Viz_state& state, int width, int height)
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

    void update_slice_extent(Viz_state& state)
    {
        state.slice_index = std::max(0, state.slice_index);
        state.slice_axis = std::clamp(state.slice_axis, 0, 2);

        int extent[6] = {
                0, state.dims[0]-1,
                0, state.dims[1]-1,
                0, state.dims[2]-1};

        const int axis_size = std::max(1, state.dims[state.slice_axis]);
        state.slice_index = std::min(state.slice_index, axis_size-1);

        if (state.slice_axis == static_cast<int>(Slice_axis::X))
            extent[0] = extent[1] = state.slice_index;
        else if (state.slice_axis == static_cast<int>(Slice_axis::Y))
            extent[2] = extent[3] = state.slice_index;
        else
            extent[4] = extent[5] = state.slice_index;

        state.slice->SetExtent(extent);
        state.slice->Modified();
    }

    Visualization_method current_method(Viz_state& state)
    {
        state.method = std::clamp(state.method, 0, 3);
        return static_cast<Visualization_method>(state.method);
    }

    std::size_t point_id(const int nx, const int ny, const int i, const int j, const int k)
    {
        return static_cast<std::size_t>((k*ny + j)*nx + i);
    }

    void update_visibility(Viz_state& state)
    {
        const auto method = current_method(state);
        const bool has_velocity = state.vector_data && state.vector_data->GetNumberOfPoints() > 0;

        state.slice_actor->SetVisibility(method == Visualization_method::Scalar_slice ? 1 : 0);
        state.volume_actor->SetVisibility(method == Visualization_method::Scalar_volume ? 1 : 0);
        state.vector_actor->SetVisibility(
                method == Visualization_method::Velocity_glyphs && has_velocity ? 1 : 0);
        state.stream_actor->SetVisibility(
                method == Visualization_method::Velocity_streamlines && has_velocity ? 1 : 0);
    }

    void update_volume_transfer(Viz_state& state)
    {
        const double min_value = state.scalar_min;
        const double max_value = state.scalar_min == state.scalar_max
                ? state.scalar_max + 1.
                : state.scalar_max;
        const double mid_value = 0.5*(min_value + max_value);
        const double opacity_scale = std::clamp(static_cast<double>(state.volume_opacity_scale), 0., 10.);

        state.volume_color->RemoveAllPoints();
        state.volume_color->AddRGBPoint(min_value, 0.05, 0.18, 0.85);
        state.volume_color->AddRGBPoint(mid_value, 0.92, 0.92, 0.86);
        state.volume_color->AddRGBPoint(max_value, 0.90, 0.12, 0.08);

        state.volume_opacity->RemoveAllPoints();
        state.volume_opacity->AddPoint(min_value, 0.00);
        state.volume_opacity->AddPoint(mid_value, 0.035*opacity_scale);
        state.volume_opacity->AddPoint(max_value, 0.18*opacity_scale);
        state.volume_property->Modified();
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

    void update_scalar_image(Viz_state& state, vtkFloatArray* scalars)
    {
        state.scalar_image->SetDimensions(state.dims[0], state.dims[1], state.dims[2]);

        double bounds[6] = {0., 1., 0., 1., 0., 1.};
        state.grid->GetBounds(bounds);
        state.scalar_image->SetOrigin(bounds[0], bounds[2], bounds[4]);
        state.scalar_image->SetSpacing(
                state.dims[0] > 1 ? (bounds[1] - bounds[0]) / (state.dims[0] - 1) : 1.,
                state.dims[1] > 1 ? (bounds[3] - bounds[2]) / (state.dims[1] - 1) : 1.,
                state.dims[2] > 1 ? (bounds[5] - bounds[4]) / (state.dims[2] - 1) : 1.);
        state.scalar_image->GetPointData()->SetScalars(scalars);
        state.scalar_image->Modified();
    }

    template<typename Snapshot>
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
                        const std::size_t point_offset = 3*id;
                        const std::size_t vector_offset = 3*id;
                        const float u = snapshot.vectors[vector_offset];
                        const float v = snapshot.vectors[vector_offset+1];
                        const float w = snapshot.vectors[vector_offset+2];
                        const float mag = std::sqrt(u*u + v*v + w*w);

                        vector_points->InsertNextPoint(
                                snapshot.points[point_offset],
                                snapshot.points[point_offset+1],
                                snapshot.points[point_offset+2]);
                        vector_vectors->InsertNextTuple3(u, v, w);
                        vector_magnitude->InsertNextValue(mag);
                    }
        }

        state.vector_data->SetPoints(vector_points);
        state.vector_data->GetPointData()->SetVectors(vector_vectors);
        state.vector_data->GetPointData()->AddArray(vector_magnitude);
        state.vector_data->Modified();
    }

    template<typename Snapshot>
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
                        const std::size_t point_offset = 3*id;
                        seed_points->InsertNextPoint(
                                snapshot.points[point_offset],
                                snapshot.points[point_offset+1],
                                snapshot.points[point_offset+2]);
                    }
        }

        state.streamline_seed_data->SetPoints(seed_points);
        state.streamline_seed_data->Modified();
    }

    template<typename TF>
    void update_dataset(Viz_state& state, Model_viz<TF>& model)
    {
        if (state.scalar_names.empty())
            throw std::runtime_error("No scalar fields are available for visualization");

        state.scalar_index = std::clamp(
                state.scalar_index, 0, static_cast<int>(state.scalar_names.size())-1);

        const auto snapshot = model.snapshot(
                state.scalar_names[static_cast<std::size_t>(state.scalar_index)],
                state.stride);

        state.dims = {{snapshot.nx, snapshot.ny, snapshot.nz}};
        state.scalar_min = snapshot.scalar_min;
        state.scalar_max = snapshot.scalar_max;
        state.vector_min = snapshot.vector_min;
        state.vector_max = snapshot.vector_max;
        state.time = snapshot.time;
        state.dt = snapshot.dt;
        state.iteration = snapshot.iteration;

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
        scalars->SetName(state.scalar_names[static_cast<std::size_t>(state.scalar_index)].c_str());
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
        state.grid->Modified();
        update_scalar_image(state, scalars);
        update_velocity_polydata(state, snapshot);
        update_streamline_seeds(state, snapshot);

        const float scalar_range_min = state.scalar_min == state.scalar_max
            ? state.scalar_min - 1.f : state.scalar_min;
        const float scalar_range_max = state.scalar_min == state.scalar_max
            ? state.scalar_max + 1.f : state.scalar_max;
        state.scalar_lut->SetTableRange(scalar_range_min, scalar_range_max);
        state.scalar_lut->Build();
        state.slice_mapper->SetScalarRange(scalar_range_min, scalar_range_max);

        const float vector_range_min = state.vector_min == state.vector_max
            ? state.vector_min - 1.f : state.vector_min;
        const float vector_range_max = state.vector_min == state.vector_max
            ? state.vector_max + 1.f : state.vector_max;
        state.vector_lut->SetTableRange(vector_range_min, vector_range_max);
        state.vector_lut->Build();
        state.vector_mapper->SetScalarRange(vector_range_min, vector_range_max);
        state.stream_mapper->SetScalarRange(vector_range_min, vector_range_max);

        update_slice_extent(state);

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
                camera->OrthogonalizeViewUp();
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
                double up[3];
                camera->GetPosition(pos);
                camera->GetFocalPoint(focal);
                camera->GetViewUp(up);

                double view_dir[3] = {
                        focal[0] - pos[0],
                        focal[1] - pos[1],
                        focal[2] - pos[2]};
                const double distance = std::max(1.e-6, norm3(view_dir));
                normalize3(view_dir);
                normalize3(up);

                double right[3];
                cross3(view_dir, up, right);
                normalize3(right);

                const double dx = x - state.pan.x;
                const double dy = y - state.pan.y;
                const double scale = 0.0015 * distance;
                const double shift[3] = {
                        (-dx*right[0] + dy*up[0]) * scale,
                        (-dx*right[1] + dy*up[1]) * scale,
                        (-dx*right[2] + dy*up[2]) * scale};

                camera->SetPosition(pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]);
                camera->SetFocalPoint(focal[0] + shift[0], focal[1] + shift[1], focal[2] + shift[2]);
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
            state->running = !state->running;
        else if (key == GLFW_KEY_N || key == GLFW_KEY_RIGHT)
            state->step_requested = true;
        else if (key == GLFW_KEY_R)
            state->restart_requested = true;
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
        state.slice = vtkSmartPointer<vtkStructuredGridGeometryFilter>::New();
        state.slice_mapper = vtkSmartPointer<vtkDataSetMapper>::New();
        state.slice_actor = vtkSmartPointer<vtkActor>::New();
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
        state.scalar_image = vtkSmartPointer<vtkImageData>::New();
        state.volume_mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
        state.volume_actor = vtkSmartPointer<vtkVolume>::New();
        state.volume_property = vtkSmartPointer<vtkVolumeProperty>::New();
        state.volume_color = vtkSmartPointer<vtkColorTransferFunction>::New();
        state.volume_opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
        state.scalar_lut = vtkSmartPointer<vtkLookupTable>::New();
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

        state.scalar_lut->SetHueRange(0.66, 0.0);
        state.scalar_lut->SetSaturationRange(0.9, 0.85);
        state.scalar_lut->SetValueRange(0.95, 0.95);
        state.scalar_lut->Build();

        state.vector_lut->SetHueRange(0.32, 0.0);
        state.vector_lut->SetSaturationRange(0.8, 0.9);
        state.vector_lut->SetValueRange(0.9, 0.95);
        state.vector_lut->Build();

        state.slice->SetInputData(state.grid);
        state.slice_mapper->SetInputConnection(state.slice->GetOutputPort());
        state.slice_mapper->SetLookupTable(state.scalar_lut);
        state.slice_mapper->ScalarVisibilityOn();
        state.slice_actor->SetMapper(state.slice_mapper);

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

        state.volume_mapper->SetInputData(state.scalar_image);
        state.volume_property->SetColor(state.volume_color);
        state.volume_property->SetScalarOpacity(state.volume_opacity);
        state.volume_property->SetInterpolationTypeToLinear();
        state.volume_property->ShadeOff();
        state.volume_actor->SetMapper(state.volume_mapper);
        state.volume_actor->SetProperty(state.volume_property);

        state.renderer->AddActor(state.slice_actor);
        state.renderer->AddVolume(state.volume_actor);
        state.renderer->AddActor(state.outline_actor);
        state.renderer->AddActor(state.vector_actor);
        state.renderer->AddActor(state.stream_actor);
    }

    void render_gui(Viz_state& state)
    {
        ImGui::SetNextWindowPos(ImVec2(14.f, 14.f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(330.f, 0.f), ImGuiCond_FirstUseEver);
        ImGui::Begin("MicroHH Viz");

        ImGui::Text("iter %d  time %.6g  dt %.3g", state.iteration, state.time, state.dt);

        if (ImGui::Button(state.running ? "Pause" : "Run"))
            state.running = !state.running;
        ImGui::SameLine();
        if (ImGui::Button("Step"))
            state.step_requested = true;
        ImGui::SameLine();
        if (ImGui::Button("Restart"))
            state.restart_requested = true;

        const char* method_labels[] = {
                "Scalar slice",
                "Scalar volume",
                "Velocity glyphs",
                "Velocity streamlines"};
        if (ImGui::Combo("Method", &state.method, method_labels, 4))
            update_visibility(state);

        const auto method = current_method(state);
        const bool scalar_method = method == Visualization_method::Scalar_slice
                || method == Visualization_method::Scalar_volume;
        const bool glyph_method = method == Visualization_method::Velocity_glyphs;
        const bool streamline_method = method == Visualization_method::Velocity_streamlines;

        std::vector<const char*> scalar_labels;
        scalar_labels.reserve(state.scalar_names.size());
        for (const auto& name : state.scalar_names)
            scalar_labels.push_back(name.c_str());

        if (scalar_method
                && ImGui::Combo(
                    "Scalar",
                    &state.scalar_index,
                    scalar_labels.data(),
                    static_cast<int>(scalar_labels.size())))
            state.needs_dataset_update = true;

        if (ImGui::SliderInt("Grid stride", &state.stride, 1, 8))
        {
            state.needs_dataset_update = true;
            state.reset_camera = true;
        }

        if (method == Visualization_method::Scalar_slice)
        {
            const char* axes[] = {"X", "Y", "Z"};
            if (ImGui::Combo("Slice axis", &state.slice_axis, axes, 3))
                update_slice_extent(state);

            const int axis_max = std::max(0, state.dims[state.slice_axis]-1);
            if (ImGui::SliderInt("Slice", &state.slice_index, 0, axis_max))
                update_slice_extent(state);
        }

        if (method == Visualization_method::Scalar_volume
                && ImGui::SliderFloat("Volume opacity", &state.volume_opacity_scale, 0.05f, 8.f))
            update_volume_transfer(state);

        if (glyph_method && ImGui::SliderInt("Vector stride", &state.vector_stride, 1, 64))
            state.needs_dataset_update = true;

        if (glyph_method && ImGui::SliderFloat("Vector scale", &state.glyph_scale, 0.05f, 100.f))
            state.needs_dataset_update = true;

        if (streamline_method && ImGui::SliderInt("Seed stride", &state.streamline_stride, 1, 64))
            state.needs_dataset_update = true;

        if (scalar_method)
            ImGui::Text("scalar %.6g .. %.6g", state.scalar_min, state.scalar_max);
        else
            ImGui::Text("speed %.6g .. %.6g", state.vector_min, state.vector_max);

        if (!scalar_method && (!state.vector_data || state.vector_data->GetNumberOfPoints() == 0))
            ImGui::Text("velocity unavailable");

        ImGui::End();
    }
}

template<typename TF>
void run_visualizer(Model_viz<TF>& model)
{
    Viz_state state;
    state.scalar_names = model.scalar_names();
    if (state.scalar_names.empty())
        throw std::runtime_error("No scalar fields are available for visualization");

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    state.window = glfwCreateWindow(1280, 800, "MicroHH Viz", nullptr, nullptr);
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
    update_dataset(state, model);

    while (!glfwWindowShouldClose(state.window))
    {
        glfwPollEvents();

        if (state.restart_requested)
        {
            state.running = false;
            model.restart();
            state.reset_camera = true;
            state.needs_dataset_update = true;
            state.restart_requested = false;
        }

        if (state.running || state.step_requested)
        {
            model.advance_one_step();
            state.needs_dataset_update = true;
            state.step_requested = false;
        }

        if (state.needs_dataset_update)
            update_dataset(state, model);

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
        render_gui(state);
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

#ifdef FLOAT_SINGLE
template void run_visualizer<float>(Model_viz<float>&);
#else
template void run_visualizer<double>(Model_viz<double>&);
#endif
