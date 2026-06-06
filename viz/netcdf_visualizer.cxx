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
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <netcdf.h>

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

namespace
{
    namespace fs = std::filesystem;

    enum class Vector_mode { Glyphs = 0, Streamlines = 1 };

    struct Snapshot
    {
        int nx = 1;
        int ny = 1;
        int nz = 1;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
        std::vector<float> points;
        std::vector<float> scalars;
        std::vector<float> vectors;
        float scalar_min = 0.f;
        float scalar_max = 0.f;
        float vector_min = 0.f;
        float vector_max = 0.f;
        double time = 0.;
        int time_index = 0;
    };

    struct Camera_drag
    {
        bool active = false;
        double x = 0.;
        double y = 0.;
    };

    struct Case_settings
    {
        fs::path ini_path;
        bool has_domain_size = false;
        bool has_grid_size = false;
        bool has_cell_size = false;
        std::array<double, 3> domain_size = {{1., 1., 1.}};
        std::array<double, 3> grid_size = {{1., 1., 1.}};
        std::array<double, 3> cell_size = {{1., 1., 1.}};
        std::array<float, 3> display_scale = {{1.f, 1.f, 1.f}};
    };

    void nc_check(const int status, const std::string& context)
    {
        if (status != NC_NOERR)
            throw std::runtime_error(context + ": " + nc_strerror(status));
    }

    bool is_x_dim(const std::string& name)
    {
        return name == "x" || name == "xh";
    }

    bool is_y_dim(const std::string& name)
    {
        return name == "y" || name == "yh";
    }

    bool is_z_dim(const std::string& name)
    {
        return name == "z" || name == "zh";
    }

    std::vector<float> index_axis(const std::size_t n)
    {
        std::vector<float> values(n);
        for (std::size_t i=0; i<n; ++i)
            values[i] = static_cast<float>(i);
        return values;
    }

    std::size_t point_id(const int nx, const int ny, const int i, const int j, const int k)
    {
        return static_cast<std::size_t>((k*ny + j)*nx + i);
    }

    std::string trim(std::string value)
    {
        const auto is_not_space = [](const unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
        value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
        return value;
    }

    std::string lower_ascii(std::string value)
    {
        std::transform(
                value.begin(), value.end(), value.begin(),
                [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::string strip_inline_comment(const std::string& line)
    {
        const auto hash = line.find('#');
        const auto semicolon = line.find(';');
        const auto cut = std::min(
                hash == std::string::npos ? line.size() : hash,
                semicolon == std::string::npos ? line.size() : semicolon);
        return line.substr(0, cut);
    }

    bool parse_double(const std::string& value, double& out)
    {
        try
        {
            std::size_t parsed = 0;
            out = std::stod(value, &parsed);
            return parsed > 0;
        }
        catch (...)
        {
            return false;
        }
    }

    std::array<float, 3> normalized_display_scale(const std::array<double, 3>& size)
    {
        const double max_size = std::max({size[0], size[1], size[2]});
        std::array<float, 3> scale = {{1.f, 1.f, 1.f}};

        for (std::size_t i=0; i<scale.size(); ++i)
        {
            if (size[i] > 0. && std::isfinite(size[i]) && max_size > 0. && std::isfinite(max_size))
                scale[i] = static_cast<float>(std::clamp(size[i] / max_size, 0.01, 1000.));
        }

        return scale;
    }

    Case_settings read_case_settings(const fs::path& directory)
    {
        Case_settings settings;
        if (directory.empty() || !fs::is_directory(directory))
            return settings;

        std::vector<fs::path> ini_files;
        for (const auto& entry : fs::directory_iterator(directory))
            if (entry.is_regular_file() && entry.path().extension() == ".ini")
                ini_files.push_back(entry.path());

        if (ini_files.empty())
            return settings;

        fs::path ini_path;
        if (ini_files.size() == 1)
            ini_path = ini_files.front();
        else
        {
            const auto preferred = directory.filename().string() + ".ini";
            const auto it = std::find_if(
                    ini_files.begin(), ini_files.end(),
                    [&](const fs::path& path) { return path.filename() == preferred; });
            if (it != ini_files.end())
                ini_path = *it;
        }

        if (ini_path.empty())
            return settings;

        std::ifstream input(ini_path);
        if (!input)
            return settings;

        settings.ini_path = ini_path;
        std::string section;
        bool have_x = false;
        bool have_y = false;
        bool have_z = false;
        bool have_itot = false;
        bool have_jtot = false;
        bool have_ktot = false;

        for (std::string line; std::getline(input, line);)
        {
            line = trim(strip_inline_comment(line));
            if (line.empty())
                continue;

            if (line.front() == '[' && line.back() == ']')
            {
                section = lower_ascii(trim(line.substr(1, line.size() - 2)));
                continue;
            }

            if (section != "grid")
                continue;

            const auto equal = line.find('=');
            if (equal == std::string::npos)
                continue;

            const auto key = lower_ascii(trim(line.substr(0, equal)));
            const auto value = trim(line.substr(equal + 1));

            double parsed = 0.;
            if (!parse_double(value, parsed) || parsed <= 0.)
                continue;

            if (key == "xsize")
            {
                settings.domain_size[0] = parsed;
                have_x = true;
            }
            else if (key == "ysize")
            {
                settings.domain_size[1] = parsed;
                have_y = true;
            }
            else if (key == "zsize")
            {
                settings.domain_size[2] = parsed;
                have_z = true;
            }
            else if (key == "itot")
            {
                settings.grid_size[0] = parsed;
                have_itot = true;
            }
            else if (key == "jtot")
            {
                settings.grid_size[1] = parsed;
                have_jtot = true;
            }
            else if (key == "ktot")
            {
                settings.grid_size[2] = parsed;
                have_ktot = true;
            }
        }

        settings.has_domain_size = have_x && have_y && have_z;
        settings.has_grid_size = have_itot && have_jtot && have_ktot;
        settings.has_cell_size = settings.has_domain_size && settings.has_grid_size;
        if (settings.has_cell_size)
        {
            for (std::size_t i=0; i<settings.cell_size.size(); ++i)
                settings.cell_size[i] = settings.domain_size[i] / settings.grid_size[i];
            settings.display_scale = normalized_display_scale(settings.cell_size);
        }
        else if (settings.has_domain_size)
            settings.display_scale = normalized_display_scale(settings.domain_size);

        return settings;
    }

    struct Field_file
    {
        std::string name;
        fs::path path;
        int ncid = -1;
        int varid = -1;
        int time_varid = -1;
        int time_dim = -1;
        int x_dim = -1;
        int y_dim = -1;
        int z_dim = -1;
        std::vector<int> dimids;
        std::vector<std::string> dim_names;
        std::vector<std::size_t> dim_lengths;
        std::vector<double> times;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;

        Field_file() = default;

        Field_file(const Field_file&) = delete;
        Field_file& operator=(const Field_file&) = delete;

        Field_file(Field_file&& other) noexcept
        {
            *this = std::move(other);
        }

        Field_file& operator=(Field_file&& other) noexcept
        {
            if (this != &other)
            {
                close();
                name = std::move(other.name);
                path = std::move(other.path);
                ncid = other.ncid;
                varid = other.varid;
                time_varid = other.time_varid;
                time_dim = other.time_dim;
                x_dim = other.x_dim;
                y_dim = other.y_dim;
                z_dim = other.z_dim;
                dimids = std::move(other.dimids);
                dim_names = std::move(other.dim_names);
                dim_lengths = std::move(other.dim_lengths);
                times = std::move(other.times);
                x = std::move(other.x);
                y = std::move(other.y);
                z = std::move(other.z);
                other.ncid = -1;
            }
            return *this;
        }

        ~Field_file()
        {
            close();
        }

        void close()
        {
            if (ncid >= 0)
            {
                nc_close(ncid);
                ncid = -1;
            }
        }

        std::size_t nx() const { return dim_lengths.at(static_cast<std::size_t>(x_dim)); }
        std::size_t ny() const { return dim_lengths.at(static_cast<std::size_t>(y_dim)); }
        std::size_t nz() const { return dim_lengths.at(static_cast<std::size_t>(z_dim)); }
        std::size_t nt() const { return time_dim >= 0 ? dim_lengths.at(static_cast<std::size_t>(time_dim)) : 1; }

        std::vector<float> read_time_slice(const std::size_t time_index) const
        {
            std::vector<std::size_t> start(dimids.size(), 0);
            std::vector<std::size_t> count = dim_lengths;
            if (time_dim >= 0)
            {
                start[static_cast<std::size_t>(time_dim)] = std::min(time_index, nt()-1);
                count[static_cast<std::size_t>(time_dim)] = 1;
            }

            std::size_t nvalues = 1;
            for (std::size_t dim=0; dim<count.size(); ++dim)
                if (static_cast<int>(dim) != time_dim)
                    nvalues *= count[dim];

            std::vector<float> values(nvalues);
            nc_check(
                    nc_get_vara_float(ncid, varid, start.data(), count.data(), values.data()),
                    "read " + name + " from " + path.string());
            return values;
        }

        double time_value(const std::size_t time_index) const
        {
            if (times.empty())
                return static_cast<double>(time_index);
            return times[std::min(time_index, times.size()-1)];
        }
    };

    std::string var_name_from_path(const fs::path& path)
    {
        auto stem = path.stem().string();
        const auto dot = stem.find('.');
        if (dot != std::string::npos)
            stem = stem.substr(0, dot);
        return stem;
    }

    int choose_data_var(const int ncid, const std::string& preferred)
    {
        int varid = -1;
        if (!preferred.empty() && nc_inq_varid(ncid, preferred.c_str(), &varid) == NC_NOERR)
            return varid;

        int nvars = 0;
        nc_check(nc_inq_nvars(ncid, &nvars), "query variable count");
        for (int id=0; id<nvars; ++id)
        {
            char name[NC_MAX_NAME + 1] = {};
            nc_type xtype = NC_NAT;
            int ndims = 0;
            nc_check(nc_inq_var(ncid, id, name, &xtype, &ndims, nullptr, nullptr), "query variable");
            if (ndims == 4 && (xtype == NC_FLOAT || xtype == NC_DOUBLE))
                return id;
        }

        throw std::runtime_error("No 4D floating-point variable found");
    }

    std::vector<float> read_coordinate(const int ncid, const std::string& dim_name, const std::size_t n)
    {
        int varid = -1;
        if (nc_inq_varid(ncid, dim_name.c_str(), &varid) != NC_NOERR)
            return index_axis(n);

        std::vector<float> values(n);
        nc_check(nc_get_var_float(ncid, varid, values.data()), "read coordinate " + dim_name);
        return values;
    }

    Field_file open_field(const fs::path& path)
    {
        Field_file field;
        field.path = path;
        const auto preferred = var_name_from_path(path);

        nc_check(nc_open(path.string().c_str(), NC_NOWRITE, &field.ncid), "open " + path.string());
        field.varid = choose_data_var(field.ncid, preferred);

        char var_name[NC_MAX_NAME + 1] = {};
        nc_type xtype = NC_NAT;
        int ndims = 0;
        nc_check(
                nc_inq_var(field.ncid, field.varid, var_name, &xtype, &ndims, nullptr, nullptr),
                "query data variable in " + path.string());

        if (ndims != 4)
            throw std::runtime_error(path.string() + " does not contain a 4D time-varying field");

        field.name = var_name;
        field.dimids.resize(static_cast<std::size_t>(ndims));
        nc_check(
                nc_inq_vardimid(field.ncid, field.varid, field.dimids.data()),
                "query dimensions for " + field.name);

        field.dim_names.resize(field.dimids.size());
        field.dim_lengths.resize(field.dimids.size());
        for (std::size_t dim=0; dim<field.dimids.size(); ++dim)
        {
            char dim_name[NC_MAX_NAME + 1] = {};
            std::size_t dim_length = 0;
            nc_check(nc_inq_dim(field.ncid, field.dimids[dim], dim_name, &dim_length), "query dimension");
            field.dim_names[dim] = dim_name;
            field.dim_lengths[dim] = dim_length;

            if (field.dim_names[dim] == "time")
                field.time_dim = static_cast<int>(dim);
            else if (is_x_dim(field.dim_names[dim]))
                field.x_dim = static_cast<int>(dim);
            else if (is_y_dim(field.dim_names[dim]))
                field.y_dim = static_cast<int>(dim);
            else if (is_z_dim(field.dim_names[dim]))
                field.z_dim = static_cast<int>(dim);
        }

        if (field.time_dim < 0 || field.x_dim < 0 || field.y_dim < 0 || field.z_dim < 0)
            throw std::runtime_error(
                    path.string() + " must use time plus x/xh, y/yh, and z/zh dimensions");

        if (nc_inq_varid(field.ncid, "time", &field.time_varid) == NC_NOERR)
        {
            field.times.resize(field.nt());
            nc_check(
                    nc_get_var_double(field.ncid, field.time_varid, field.times.data()),
                    "read time from " + path.string());
        }

        field.x = read_coordinate(field.ncid, field.dim_names[static_cast<std::size_t>(field.x_dim)], field.nx());
        field.y = read_coordinate(field.ncid, field.dim_names[static_cast<std::size_t>(field.y_dim)], field.ny());
        field.z = read_coordinate(field.ncid, field.dim_names[static_cast<std::size_t>(field.z_dim)], field.nz());
        return field;
    }

    struct Axis_sample
    {
        int lo = 0;
        int hi = 0;
        float weight = 0.f;
    };

    Axis_sample locate_axis_sample(const std::vector<float>& axis, const float value)
    {
        if (axis.size() <= 1)
            return {};

        const bool ascending = axis.front() <= axis.back();
        if (ascending)
        {
            if (value <= axis.front())
                return {0, 0, 0.f};
            if (value >= axis.back())
            {
                const int last = static_cast<int>(axis.size()) - 1;
                return {last, last, 0.f};
            }

            const auto hi_it = std::upper_bound(axis.begin(), axis.end(), value);
            const int hi = static_cast<int>(std::distance(axis.begin(), hi_it));
            const int lo = hi - 1;
            const float denom = axis[static_cast<std::size_t>(hi)] - axis[static_cast<std::size_t>(lo)];
            const float weight = denom == 0.f ? 0.f : (value - axis[static_cast<std::size_t>(lo)]) / denom;
            return {lo, hi, std::clamp(weight, 0.f, 1.f)};
        }

        if (value >= axis.front())
            return {0, 0, 0.f};
        if (value <= axis.back())
        {
            const int last = static_cast<int>(axis.size()) - 1;
            return {last, last, 0.f};
        }

        for (int hi=1; hi<static_cast<int>(axis.size()); ++hi)
        {
            const int lo = hi - 1;
            if (value <= axis[static_cast<std::size_t>(lo)] && value >= axis[static_cast<std::size_t>(hi)])
            {
                const float denom = axis[static_cast<std::size_t>(hi)] - axis[static_cast<std::size_t>(lo)];
                const float weight = denom == 0.f ? 0.f : (value - axis[static_cast<std::size_t>(lo)]) / denom;
                return {lo, hi, std::clamp(weight, 0.f, 1.f)};
            }
        }

        return {};
    }

    std::vector<Axis_sample> map_axis_samples(
            const std::vector<float>& target_axis,
            const std::vector<float>& source_axis)
    {
        std::vector<Axis_sample> samples;
        samples.reserve(target_axis.size());
        for (const auto value : target_axis)
            samples.push_back(locate_axis_sample(source_axis, value));
        return samples;
    }

    float lerp(const float a, const float b, const float weight)
    {
        return a + weight*(b - a);
    }

    float sample_component(
            const std::vector<float>& values,
            const int nx,
            const int ny,
            const Axis_sample& x,
            const Axis_sample& y,
            const Axis_sample& z)
    {
        const float c000 = values[point_id(nx, ny, x.lo, y.lo, z.lo)];
        const float c100 = values[point_id(nx, ny, x.hi, y.lo, z.lo)];
        const float c010 = values[point_id(nx, ny, x.lo, y.hi, z.lo)];
        const float c110 = values[point_id(nx, ny, x.hi, y.hi, z.lo)];
        const float c001 = values[point_id(nx, ny, x.lo, y.lo, z.hi)];
        const float c101 = values[point_id(nx, ny, x.hi, y.lo, z.hi)];
        const float c011 = values[point_id(nx, ny, x.lo, y.hi, z.hi)];
        const float c111 = values[point_id(nx, ny, x.hi, y.hi, z.hi)];

        const float c00 = lerp(c000, c100, x.weight);
        const float c10 = lerp(c010, c110, x.weight);
        const float c01 = lerp(c001, c101, x.weight);
        const float c11 = lerp(c011, c111, x.weight);
        const float c0 = lerp(c00, c10, y.weight);
        const float c1 = lerp(c01, c11, y.weight);
        return lerp(c0, c1, z.weight);
    }

    void update_minmax(const float value, float& min_value, float& max_value)
    {
        if (!std::isfinite(value))
            return;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    class Dataset
    {
        public:
            explicit Dataset(const std::vector<fs::path>& paths)
            {
                for (const auto& path : paths)
                {
                    if (!fs::exists(path))
                        continue;

                    auto field = open_field(path);
                    if (field.name == "u" || field.name == "v" || field.name == "w")
                        velocity_.emplace(field.name, std::move(field));
                    else
                    {
                        scalar_names_.push_back(field.name);
                        scalars_.emplace(field.name, std::move(field));
                    }
                }

                std::sort(scalar_names_.begin(), scalar_names_.end());
                if (scalars_.count("ql") && scalars_.count("qi"))
                    scalar_names_.push_back("ql+qi");
                if (scalar_names_.empty())
                    throw std::runtime_error("No scalar NetCDF dumps were loaded. Expected files such as ql.nc or qi.nc.");
            }

            const std::vector<std::string>& scalar_names() const
            {
                return scalar_names_;
            }

            bool has_velocity() const
            {
                return !velocity_.empty();
            }

            int time_count(const std::string& scalar_name) const
            {
                if (scalar_name == "ql+qi")
                    return static_cast<int>(std::min(scalars_.at("ql").nt(), scalars_.at("qi").nt()));
                return static_cast<int>(scalars_.at(scalar_name).nt());
            }

            Snapshot snapshot(
                    const std::string& scalar_name,
                    const int time_index,
                    const int stride,
                    const bool include_velocity) const
            {
                const bool combined_scalar = scalar_name == "ql+qi";
                const auto& scalar = combined_scalar ? scalars_.at("ql") : scalars_.at(scalar_name);
                const int source_nx = static_cast<int>(scalar.nx());
                const int source_ny = static_cast<int>(scalar.ny());
                const int source_nz = static_cast<int>(scalar.nz());
                const int s = std::max(1, stride);

                Snapshot snapshot;
                snapshot.nx = (source_nx + s - 1) / s;
                snapshot.ny = (source_ny + s - 1) / s;
                snapshot.nz = (source_nz + s - 1) / s;
                snapshot.x.resize(static_cast<std::size_t>(snapshot.nx));
                snapshot.y.resize(static_cast<std::size_t>(snapshot.ny));
                snapshot.z.resize(static_cast<std::size_t>(snapshot.nz));
                snapshot.time_index = std::clamp(time_index, 0, static_cast<int>(scalar.nt()) - 1);
                snapshot.time = scalar.time_value(static_cast<std::size_t>(snapshot.time_index));

                auto scalar_values = scalar.read_time_slice(static_cast<std::size_t>(snapshot.time_index));
                if (combined_scalar)
                {
                    const auto qi_values = scalars_.at("qi").read_time_slice(static_cast<std::size_t>(snapshot.time_index));
                    if (qi_values.size() != scalar_values.size())
                        throw std::runtime_error("ql+qi requires ql and qi to have identical dimensions");
                    for (std::size_t id=0; id<scalar_values.size(); ++id)
                        scalar_values[id] += qi_values[id];
                }

                std::map<std::string, std::vector<float>> velocity_values;
                if (include_velocity)
                    for (const auto& [name, field] : velocity_)
                        velocity_values.emplace(name, field.read_time_slice(
                                std::min<std::size_t>(static_cast<std::size_t>(snapshot.time_index), field.nt()-1)));

                const std::size_t npoints = static_cast<std::size_t>(snapshot.nx) * snapshot.ny * snapshot.nz;
                snapshot.points.resize(3*npoints);
                snapshot.scalars.resize(npoints);
                if (!velocity_values.empty())
                    snapshot.vectors.assign(3*npoints, 0.f);

                snapshot.scalar_min = std::numeric_limits<float>::max();
                snapshot.scalar_max = -std::numeric_limits<float>::max();
                snapshot.vector_min = std::numeric_limits<float>::max();
                snapshot.vector_max = -std::numeric_limits<float>::max();

                for (int i=0; i<snapshot.nx; ++i)
                    snapshot.x[static_cast<std::size_t>(i)] =
                        scalar.x[static_cast<std::size_t>(std::min(i*s, source_nx - 1))];
                for (int j=0; j<snapshot.ny; ++j)
                    snapshot.y[static_cast<std::size_t>(j)] =
                        scalar.y[static_cast<std::size_t>(std::min(j*s, source_ny - 1))];
                for (int k=0; k<snapshot.nz; ++k)
                    snapshot.z[static_cast<std::size_t>(k)] =
                        scalar.z[static_cast<std::size_t>(std::min(k*s, source_nz - 1))];

                struct Velocity_sampler
                {
                    const Field_file* field = nullptr;
                    const std::vector<float>* values = nullptr;
                    std::vector<Axis_sample> x;
                    std::vector<Axis_sample> y;
                    std::vector<Axis_sample> z;
                };

                std::map<std::string, Velocity_sampler> velocity_samplers;
                for (const auto& [name, values] : velocity_values)
                {
                    const auto& field = velocity_.at(name);
                    velocity_samplers.emplace(
                            name,
                            Velocity_sampler{
                                &field,
                                &values,
                                map_axis_samples(snapshot.x, field.x),
                                map_axis_samples(snapshot.y, field.y),
                                map_axis_samples(snapshot.z, field.z)});
                }

                for (int k=0; k<snapshot.nz; ++k)
                    for (int j=0; j<snapshot.ny; ++j)
                        for (int i=0; i<snapshot.nx; ++i)
                        {
                            const int source_i = std::min(i*s, source_nx - 1);
                            const int source_j = std::min(j*s, source_ny - 1);
                            const int source_k = std::min(k*s, source_nz - 1);
                            const auto id = point_id(snapshot.nx, snapshot.ny, i, j, k);
                            const auto source_id = point_id(source_nx, source_ny, source_i, source_j, source_k);

                            snapshot.points[3*id] = snapshot.x[static_cast<std::size_t>(i)];
                            snapshot.points[3*id + 1] = snapshot.y[static_cast<std::size_t>(j)];
                            snapshot.points[3*id + 2] = snapshot.z[static_cast<std::size_t>(k)];
                            snapshot.scalars[id] = scalar_values[source_id];
                            update_minmax(snapshot.scalars[id], snapshot.scalar_min, snapshot.scalar_max);

                            if (!snapshot.vectors.empty())
                            {
                                float u = 0.f;
                                float v = 0.f;
                                float w = 0.f;

                                if (const auto it = velocity_samplers.find("u"); it != velocity_samplers.end())
                                    u = sample_component(
                                            *it->second.values,
                                            static_cast<int>(it->second.field->nx()),
                                            static_cast<int>(it->second.field->ny()),
                                            it->second.x[static_cast<std::size_t>(i)],
                                            it->second.y[static_cast<std::size_t>(j)],
                                            it->second.z[static_cast<std::size_t>(k)]);
                                if (const auto it = velocity_samplers.find("v"); it != velocity_samplers.end())
                                    v = sample_component(
                                            *it->second.values,
                                            static_cast<int>(it->second.field->nx()),
                                            static_cast<int>(it->second.field->ny()),
                                            it->second.x[static_cast<std::size_t>(i)],
                                            it->second.y[static_cast<std::size_t>(j)],
                                            it->second.z[static_cast<std::size_t>(k)]);
                                if (const auto it = velocity_samplers.find("w"); it != velocity_samplers.end())
                                    w = sample_component(
                                            *it->second.values,
                                            static_cast<int>(it->second.field->nx()),
                                            static_cast<int>(it->second.field->ny()),
                                            it->second.x[static_cast<std::size_t>(i)],
                                            it->second.y[static_cast<std::size_t>(j)],
                                            it->second.z[static_cast<std::size_t>(k)]);

                                snapshot.vectors[3*id] = u;
                                snapshot.vectors[3*id + 1] = v;
                                snapshot.vectors[3*id + 2] = w;
                                const float speed = std::sqrt(u*u + v*v + w*w);
                                update_minmax(speed, snapshot.vector_min, snapshot.vector_max);
                            }
                        }

                if (snapshot.scalar_min == std::numeric_limits<float>::max())
                    snapshot.scalar_min = snapshot.scalar_max = 0.f;
                if (snapshot.vector_min == std::numeric_limits<float>::max())
                    snapshot.vector_min = snapshot.vector_max = 0.f;

                return snapshot;
            }

        private:
            std::map<std::string, Field_file> scalars_;
            std::map<std::string, Field_file> velocity_;
            std::vector<std::string> scalar_names_;
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
        vtkSmartPointer<vtkLookupTable> scalar_lut;
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
        bool reset_camera = true;
        bool volume_resampled = false;
        float glyph_scale = 5.f;
        float volume_opacity_scale = 1.f;
        float playback_rate = 8.f;
        std::array<float, 3> display_scale = {{1.f, 1.f, 1.f}};

        std::array<int, 3> dims = {{1, 1, 1}};
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

    void update_actor_scale(Viz_state& state)
    {
        const double xscale = std::max(0.01f, state.display_scale[0]);
        const double yscale = std::max(0.01f, state.display_scale[1]);
        const double zscale = std::max(0.01f, state.display_scale[2]);
        state.outline_actor->SetScale(xscale, yscale, zscale);
        state.vector_actor->SetScale(xscale, yscale, zscale);
        state.stream_actor->SetScale(xscale, yscale, zscale);
        state.volume_actor->SetScale(xscale, yscale, zscale);
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

    double uniform_spacing(const std::vector<float>& axis)
    {
        if (axis.size() <= 1)
            return 1.;
        return static_cast<double>(axis.back() - axis.front()) / static_cast<double>(axis.size() - 1);
    }

    bool axis_is_uniform(const std::vector<float>& axis)
    {
        if (axis.size() <= 2)
            return true;

        const double reference = uniform_spacing(axis);
        const double tolerance = std::max(1.e-5, std::abs(reference) * 1.e-4);
        for (std::size_t i=1; i<axis.size(); ++i)
            if (std::abs(static_cast<double>(axis[i] - axis[i-1]) - reference) > tolerance)
                return false;

        return true;
    }

    vtkSmartPointer<vtkFloatArray> resample_scalars_to_uniform_image(
            const Snapshot& snapshot,
            vtkFloatArray* source_scalars)
    {
        auto values = vtkSmartPointer<vtkFloatArray>::New();
        values->SetName(source_scalars->GetName());
        values->SetNumberOfComponents(1);
        values->SetNumberOfTuples(static_cast<vtkIdType>(snapshot.scalars.size()));

        std::vector<Axis_sample> x_samples;
        std::vector<Axis_sample> y_samples;
        std::vector<Axis_sample> z_samples;
        x_samples.reserve(static_cast<std::size_t>(snapshot.nx));
        y_samples.reserve(static_cast<std::size_t>(snapshot.ny));
        z_samples.reserve(static_cast<std::size_t>(snapshot.nz));

        for (int i=0; i<snapshot.nx; ++i)
        {
            const float x = snapshot.nx <= 1
                    ? (snapshot.x.empty() ? 0.f : snapshot.x.front())
                    : snapshot.x.front() + static_cast<float>(i) *
                        (snapshot.x.back() - snapshot.x.front()) / static_cast<float>(snapshot.nx - 1);
            x_samples.push_back(locate_axis_sample(snapshot.x, x));
        }
        for (int j=0; j<snapshot.ny; ++j)
        {
            const float y = snapshot.ny <= 1
                    ? (snapshot.y.empty() ? 0.f : snapshot.y.front())
                    : snapshot.y.front() + static_cast<float>(j) *
                        (snapshot.y.back() - snapshot.y.front()) / static_cast<float>(snapshot.ny - 1);
            y_samples.push_back(locate_axis_sample(snapshot.y, y));
        }
        for (int k=0; k<snapshot.nz; ++k)
        {
            const float z = snapshot.nz <= 1
                    ? (snapshot.z.empty() ? 0.f : snapshot.z.front())
                    : snapshot.z.front() + static_cast<float>(k) *
                        (snapshot.z.back() - snapshot.z.front()) / static_cast<float>(snapshot.nz - 1);
            z_samples.push_back(locate_axis_sample(snapshot.z, z));
        }

        for (int k=0; k<snapshot.nz; ++k)
            for (int j=0; j<snapshot.ny; ++j)
                for (int i=0; i<snapshot.nx; ++i)
                {
                    const auto id = point_id(snapshot.nx, snapshot.ny, i, j, k);
                    values->SetValue(
                            static_cast<vtkIdType>(id),
                            sample_component(
                                snapshot.scalars,
                                snapshot.nx,
                                snapshot.ny,
                                x_samples[static_cast<std::size_t>(i)],
                                y_samples[static_cast<std::size_t>(j)],
                                z_samples[static_cast<std::size_t>(k)]));
                }

        return values;
    }

    void update_volume_image(Viz_state& state, const Snapshot& snapshot, vtkFloatArray* scalars)
    {
        state.volume_image->SetDimensions(snapshot.nx, snapshot.ny, snapshot.nz);
        state.volume_image->SetOrigin(
                snapshot.x.empty() ? 0. : snapshot.x.front(),
                snapshot.y.empty() ? 0. : snapshot.y.front(),
                snapshot.z.empty() ? 0. : snapshot.z.front());
        state.volume_image->SetSpacing(
                uniform_spacing(snapshot.x),
                uniform_spacing(snapshot.y),
                uniform_spacing(snapshot.z));

        state.volume_resampled = !axis_is_uniform(snapshot.x)
                || !axis_is_uniform(snapshot.y)
                || !axis_is_uniform(snapshot.z);

        if (state.volume_resampled)
            state.volume_image->GetPointData()->SetScalars(
                    resample_scalars_to_uniform_image(snapshot, scalars));
        else
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
                scalar_name, state.time_index, state.stride, include_velocity);
        state.dims = {{snapshot.nx, snapshot.ny, snapshot.nz}};
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

        const float scalar_range_min = state.scalar_min == state.scalar_max
            ? state.scalar_min - 1.f : state.scalar_min;
        const float scalar_range_max = state.scalar_min == state.scalar_max
            ? state.scalar_max + 1.f : state.scalar_max;
        state.scalar_lut->SetTableRange(scalar_range_min, scalar_range_max);
        state.scalar_lut->Build();

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
        update_actor_scale(state);
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
        state.volume_property->SetColor(state.volume_color);
        state.volume_property->SetScalarOpacity(state.volume_opacity);
        state.volume_property->SetInterpolationTypeToLinear();
        state.volume_property->ShadeOff();
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
            if (state.case_settings.has_cell_size)
                ImGui::Text(
                        "cell %.6g x %.6g x %.6g m",
                        state.case_settings.cell_size[0],
                        state.case_settings.cell_size[1],
                        state.case_settings.cell_size[2]);
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

        if (ImGui::SliderInt("Frame", &state.time_index, 0, state.max_time_index))
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

        if (ImGui::SliderInt("Grid stride", &state.stride, 1, 16))
        {
            state.needs_dataset_update = true;
            state.reset_camera = true;
        }

        if (state.show_scalar
                && ImGui::SliderFloat("Volume opacity", &state.volume_opacity_scale, 0.05f, 8.f))
            update_volume_transfer(state);

        bool scale_changed = false;
        scale_changed |= ImGui::SliderFloat("X scale", &state.display_scale[0], 0.01f, 50.f, "%.2f");
        scale_changed |= ImGui::SliderFloat("Y scale", &state.display_scale[1], 0.01f, 50.f, "%.2f");
        scale_changed |= ImGui::SliderFloat("Z scale", &state.display_scale[2], 0.01f, 50.f, "%.2f");
        if ((state.case_settings.has_cell_size || state.case_settings.has_domain_size)
                && ImGui::Button("Auto scale"))
        {
            state.display_scale = state.case_settings.display_scale;
            scale_changed = true;
        }
        if (scale_changed)
        {
            update_actor_scale(state);
            state.reset_camera = true;
        }

        if (glyph_method && ImGui::SliderInt("Vector stride", &state.vector_stride, 1, 64))
            state.needs_dataset_update = true;

        if (glyph_method && ImGui::SliderFloat("Vector scale", &state.glyph_scale, 0.05f, 100.f))
            state.vector_mapper->SetScaleFactor(state.glyph_scale * state.max_extent * 0.04);

        if (streamline_method && ImGui::SliderInt("Seed stride", &state.streamline_stride, 1, 64))
            state.needs_dataset_update = true;

        if (state.show_scalar)
            ImGui::Text("scalar %.6g .. %.6g", state.scalar_min, state.scalar_max);

        if (state.show_vectors)
            ImGui::Text("speed %.6g .. %.6g", state.vector_min, state.vector_max);

        if (state.show_vectors && (!dataset.has_velocity() || !state.vector_data || state.vector_data->GetNumberOfPoints() == 0))
            ImGui::Text("velocity unavailable");

        ImGui::End();
    }

    std::vector<fs::path> discover_paths(const std::vector<std::string>& args)
    {
        std::vector<fs::path> paths;
        fs::path directory;

        for (std::size_t i=0; i<args.size(); ++i)
        {
            if ((args[i] == "-d" || args[i] == "--directory") && i + 1 < args.size())
                directory = args[++i];
            else
                paths.emplace_back(args[i]);
        }

        if (paths.size() == 1 && fs::is_directory(paths.front()))
        {
            directory = paths.front();
            paths.clear();
        }

        if (directory.empty() && paths.empty())
            directory = fs::current_path();

        if (!directory.empty())
        {
            for (const char* name : {"ql.nc", "qi.nc", "u.nc", "v.nc", "w.nc"})
            {
                auto path = directory / name;
                if (fs::exists(path))
                    paths.push_back(path);
            }
        }

        if (paths.empty())
            throw std::runtime_error("No NetCDF files found. Pass a directory containing ql.nc/qi.nc/u.nc/v.nc/w.nc or explicit .nc paths.");

        return paths;
    }

    fs::path common_parent_directory(const std::vector<fs::path>& paths)
    {
        fs::path directory;

        for (const auto& path : paths)
        {
            const auto current = fs::absolute(path).parent_path();
            if (directory.empty())
                directory = current;
            else if (directory != current)
                return {};
        }

        return directory;
    }

    void print_usage()
    {
        std::cout
            << "Usage: microhh_nc_viz [directory]\n"
            << "       microhh_nc_viz --directory CASE_DIR\n"
            << "       microhh_nc_viz ql.nc qi.nc u.nc v.nc w.nc\n\n"
            << "The viewer reads NetCDF files produced by python/3d_to_nc.py.\n";
    }

    void run_visualizer(const Dataset& dataset, const Case_settings& case_settings)
    {
        Viz_state state;
        state.scalar_names = dataset.scalar_names();
        if (const auto it = std::find(state.scalar_names.begin(), state.scalar_names.end(), "ql+qi");
                it != state.scalar_names.end())
            state.scalar_index = static_cast<int>(std::distance(state.scalar_names.begin(), it));
        state.case_settings = case_settings;
        if (state.case_settings.has_domain_size)
            state.display_scale = state.case_settings.display_scale;

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

            if (state.needs_dataset_update)
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

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string> args;
        for (int i=1; i<argc; ++i)
            args.emplace_back(argv[i]);

        if (std::find(args.begin(), args.end(), "-h") != args.end()
                || std::find(args.begin(), args.end(), "--help") != args.end())
        {
            print_usage();
            return 0;
        }

        const auto paths = discover_paths(args);
        const auto case_settings = read_case_settings(common_parent_directory(paths));
        Dataset dataset(paths);
        run_visualizer(dataset, case_settings);
    }
    catch (const std::exception& e)
    {
        std::cerr << "microhh_nc_viz: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
