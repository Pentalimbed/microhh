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

#include "netcdf_data.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include <netcdf.h>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Transform.h>

namespace microhh::viz
{
namespace
{
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
            if (size[i] > 0. && max_size > 0.)
                scale[i] = static_cast<float>(std::clamp(size[i] / max_size, 0.01, 1000.));
        }

        return scale;
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

    void update_minmax(const float value, float& min_value, float& max_value)
    {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    double average_axis_spacing(const std::vector<float>& axis)
    {
        if (axis.size() <= 1)
            return 1.;

        const double spacing =
                static_cast<double>(axis.back() - axis.front()) /
                static_cast<double>(axis.size() - 1);
        return spacing != 0. ? spacing : 1.;
    }

    float linear_axis_value(const std::vector<float>& axis, const int index)
    {
        if (axis.empty())
            return static_cast<float>(index);
        if (axis.size() <= 1)
            return axis.front();

        return axis.front()
                + static_cast<float>(index) *
                (axis.back() - axis.front()) / static_cast<float>(axis.size() - 1);
    }

    bool axis_is_regular_for_export(const std::vector<float>& axis)
    {
        if (axis.size() <= 2)
            return true;

        const double reference = average_axis_spacing(axis);
        const double tolerance = std::max(1.e-5, std::abs(reference) * 1.e-4);
        for (std::size_t i=1; i<axis.size(); ++i)
            if (std::abs(static_cast<double>(axis[i] - axis[i-1]) - reference) > tolerance)
                return false;

        return true;
    }

    bool axes_match_for_combination(
            const std::vector<float>& a,
            const std::vector<float>& b)
    {
        if (a.size() != b.size())
            return false;

        for (std::size_t i=0; i<a.size(); ++i)
        {
            const double scale = std::max({1., std::abs(static_cast<double>(a[i])), std::abs(static_cast<double>(b[i]))});
            if (std::abs(static_cast<double>(a[i] - b[i])) > 1.e-5 * scale)
                return false;
        }

        return true;
    }
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

struct Dataset::Impl
{
    std::map<std::string, Field_file> scalars;
    std::map<std::string, Field_file> velocity;
    std::vector<std::string> scalar_names;
};

Dataset::Dataset(const std::vector<fs::path>& paths)
    : impl_(std::make_unique<Impl>())
{
    for (const auto& path : paths)
    {
        if (!fs::exists(path))
            continue;

        auto field = open_field(path);
        if (field.name == "u" || field.name == "v" || field.name == "w")
            impl_->velocity.emplace(field.name, std::move(field));
        else
        {
            impl_->scalar_names.push_back(field.name);
            impl_->scalars.emplace(field.name, std::move(field));
        }
    }

    std::sort(impl_->scalar_names.begin(), impl_->scalar_names.end());
    if (impl_->scalars.count("ql") && impl_->scalars.count("qi"))
        impl_->scalar_names.push_back("ql+qi");
    if (impl_->scalar_names.empty())
        throw std::runtime_error("No scalar NetCDF dumps were loaded. Expected files such as ql.nc or qi.nc.");
}

Dataset::~Dataset() = default;

Dataset::Dataset(Dataset&&) noexcept = default;

Dataset& Dataset::operator=(Dataset&&) noexcept = default;

const std::vector<std::string>& Dataset::scalar_names() const
{
    return impl_->scalar_names;
}

bool Dataset::has_velocity() const
{
    return !impl_->velocity.empty();
}

bool Dataset::has_q_criterion_velocity() const
{
    return impl_->velocity.count("u") && impl_->velocity.count("v") && impl_->velocity.count("w");
}

bool Dataset::has_total_cloud_density() const
{
    return impl_->scalars.count("ql") && impl_->scalars.count("qi");
}

int Dataset::time_count(const std::string& scalar_name) const
{
    if (scalar_name == "ql+qi")
        return static_cast<int>(std::min(impl_->scalars.at("ql").nt(), impl_->scalars.at("qi").nt()));
    return static_cast<int>(impl_->scalars.at(scalar_name).nt());
}

Vdb_export_summary Dataset::export_total_cloud_density_vdb_sequence(const fs::path& directory) const
{
    if (!has_total_cloud_density())
        throw std::runtime_error("VDB export requires both ql.nc and qi.nc");
    if (!has_q_criterion_velocity())
        throw std::runtime_error("VDB export requires u.nc, v.nc, and w.nc");

    const auto& ql = impl_->scalars.at("ql");
    const auto& qi = impl_->scalars.at("qi");
    if (ql.nx() != qi.nx() || ql.ny() != qi.ny() || ql.nz() != qi.nz())
        throw std::runtime_error("ql and qi dimensions do not match");
    if (!axes_match_for_combination(ql.x, qi.x)
            || !axes_match_for_combination(ql.y, qi.y)
            || !axes_match_for_combination(ql.z, qi.z))
        throw std::runtime_error("ql and qi coordinate axes do not match");

    const std::size_t frame_count = std::min(ql.nt(), qi.nt());
    if (frame_count == 0)
        throw std::runtime_error("No ql/qi frames are available for VDB export");

    fs::create_directories(directory);

    const int nx = static_cast<int>(ql.nx());
    const int ny = static_cast<int>(ql.ny());
    const int nz = static_cast<int>(ql.nz());
    const bool resample_to_uniform = !axis_is_regular_for_export(ql.x)
            || !axis_is_regular_for_export(ql.y)
            || !axis_is_regular_for_export(ql.z);

    std::vector<Axis_sample> x_samples;
    std::vector<Axis_sample> y_samples;
    std::vector<Axis_sample> z_samples;
    if (resample_to_uniform)
    {
        x_samples.reserve(ql.x.size());
        y_samples.reserve(ql.y.size());
        z_samples.reserve(ql.z.size());
        for (int i=0; i<nx; ++i)
            x_samples.push_back(locate_axis_sample(ql.x, linear_axis_value(ql.x, i)));
        for (int j=0; j<ny; ++j)
            y_samples.push_back(locate_axis_sample(ql.y, linear_axis_value(ql.y, j)));
        for (int k=0; k<nz; ++k)
            z_samples.push_back(locate_axis_sample(ql.z, linear_axis_value(ql.z, k)));
    }

    Vdb_export_summary summary;
    summary.frames = frame_count;
    summary.resampled = resample_to_uniform;

    for (std::size_t frame=0; frame<frame_count; ++frame)
    {
        const auto ql_values = ql.read_time_slice(frame);
        const auto qi_values = qi.read_time_slice(frame);
        if (qi_values.size() != ql_values.size())
            throw std::runtime_error("ql and qi frame sizes do not match");

        const auto u_values = impl_->velocity.at("u").read_time_slice(std::min<std::size_t>(frame, impl_->velocity.at("u").nt() - 1));
        const auto v_values = impl_->velocity.at("v").read_time_slice(std::min<std::size_t>(frame, impl_->velocity.at("v").nt() - 1));
        const auto w_values = impl_->velocity.at("w").read_time_slice(std::min<std::size_t>(frame, impl_->velocity.at("w").nt() - 1));
        if (u_values.size() != ql_values.size() || v_values.size() != ql_values.size() || w_values.size() != ql_values.size())
            throw std::runtime_error("velocity and density frame sizes do not match");

        auto density_grid = openvdb::FloatGrid::create(0.f);
        density_grid->setName("cloud_density");
        density_grid->setGridClass(openvdb::GRID_FOG_VOLUME);

        auto q_grid = openvdb::FloatGrid::create(0.f);
        q_grid->setName("q_criterion");
        q_grid->setGridClass(openvdb::GRID_FOG_VOLUME);

        auto transform = openvdb::math::Transform::createLinearTransform(1.);
        transform->postScale(openvdb::math::Vec3d(
                    average_axis_spacing(ql.x),
                    average_axis_spacing(ql.y),
                    average_axis_spacing(ql.z)));
        transform->postTranslate(openvdb::math::Vec3d(
                    ql.x.empty() ? 0. : ql.x.front(),
                    ql.y.empty() ? 0. : ql.y.front(),
                    ql.z.empty() ? 0. : ql.z.front()));
        density_grid->setTransform(transform);
        q_grid->setTransform(transform->copy());

        density_grid->insertMeta("microhh_source", openvdb::StringMetadata("ql+qi"));
        density_grid->insertMeta("microhh_time", openvdb::DoubleMetadata(ql.time_value(frame)));
        density_grid->insertMeta("microhh_time_index", openvdb::Int32Metadata(static_cast<std::int32_t>(frame)));
        density_grid->insertMeta("microhh_resampled_to_uniform", openvdb::StringMetadata(resample_to_uniform ? "true" : "false"));
        q_grid->insertMeta("microhh_source", openvdb::StringMetadata("q_criterion"));
        q_grid->insertMeta("microhh_time", openvdb::DoubleMetadata(ql.time_value(frame)));
        q_grid->insertMeta("microhh_time_index", openvdb::Int32Metadata(static_cast<std::int32_t>(frame)));
        q_grid->insertMeta("microhh_resampled_to_uniform", openvdb::StringMetadata(resample_to_uniform ? "true" : "false"));

        auto density_accessor = density_grid->getAccessor();
        auto q_accessor = q_grid->getAccessor();
        std::size_t active_voxels = 0;
        for (int k=0; k<nz; ++k)
            for (int j=0; j<ny; ++j)
                for (int i=0; i<nx; ++i)
                {
                    const float density = resample_to_uniform
                            ? sample_component(
                                    ql_values, nx, ny,
                                    x_samples[static_cast<std::size_t>(i)],
                                    y_samples[static_cast<std::size_t>(j)],
                                    z_samples[static_cast<std::size_t>(k)])
                            + sample_component(
                                    qi_values, nx, ny,
                                    x_samples[static_cast<std::size_t>(i)],
                                    y_samples[static_cast<std::size_t>(j)],
                                    z_samples[static_cast<std::size_t>(k)])
                            : ql_values[point_id(nx, ny, i, j, k)] + qi_values[point_id(nx, ny, i, j, k)];

                    if (density <= 0.f)
                        continue;

                    const int im = std::max(0, i - 1);
                    const int ip = std::min(nx - 1, i + 1);
                    const int jm = std::max(0, j - 1);
                    const int jp = std::min(ny - 1, j + 1);
                    const int km = std::max(0, k - 1);
                    const int kp = std::min(nz - 1, k + 1);

                    const auto sample_velocity = [&](const std::vector<float>& values, const int ii, const int jj, const int kk)
                    {
                        if (resample_to_uniform)
                            return sample_component(
                                    values, nx, ny,
                                    x_samples[static_cast<std::size_t>(ii)],
                                    y_samples[static_cast<std::size_t>(jj)],
                                    z_samples[static_cast<std::size_t>(kk)]);
                        return values[point_id(nx, ny, ii, jj, kk)];
                    };

                    const auto grid_axis_value = [&](const std::vector<float>& axis, const int idx)
                    {
                        if (axis.empty())
                            return static_cast<double>(idx);
                        if (resample_to_uniform)
                            return static_cast<double>(linear_axis_value(axis, idx));
                        return static_cast<double>(axis[static_cast<std::size_t>(idx)]);
                    };

                    const auto axis_step = [](const std::vector<float>& axis, const int a, const int b)
                    {
                        return std::max(1.e-12, std::abs(static_cast<double>(axis[static_cast<std::size_t>(a)]
                                - axis[static_cast<std::size_t>(b)])));
                    };

                    const double dx = std::max(1.e-12, std::abs(grid_axis_value(ql.x, ip) - grid_axis_value(ql.x, im)));
                    const double dy = std::max(1.e-12, std::abs(grid_axis_value(ql.y, jp) - grid_axis_value(ql.y, jm)));
                    const double dz = std::max(1.e-12, std::abs(grid_axis_value(ql.z, kp) - grid_axis_value(ql.z, km)));

                    const double du_dx = (ip == im) ? 0. : (sample_velocity(u_values, ip, j, k) - sample_velocity(u_values, im, j, k)) / dx;
                    const double du_dy = (jp == jm) ? 0. : (sample_velocity(u_values, i, jp, k) - sample_velocity(u_values, i, jm, k)) / dy;
                    const double du_dz = (kp == km) ? 0. : (sample_velocity(u_values, i, j, kp) - sample_velocity(u_values, i, j, km)) / dz;

                    const double dv_dx = (ip == im) ? 0. : (sample_velocity(v_values, ip, j, k) - sample_velocity(v_values, im, j, k)) / dx;
                    const double dv_dy = (jp == jm) ? 0. : (sample_velocity(v_values, i, jp, k) - sample_velocity(v_values, i, jm, k)) / dy;
                    const double dv_dz = (kp == km) ? 0. : (sample_velocity(v_values, i, j, kp) - sample_velocity(v_values, i, j, km)) / dz;

                    const double dw_dx = (ip == im) ? 0. : (sample_velocity(w_values, ip, j, k) - sample_velocity(w_values, im, j, k)) / dx;
                    const double dw_dy = (jp == jm) ? 0. : (sample_velocity(w_values, i, jp, k) - sample_velocity(w_values, i, jm, k)) / dy;
                    const double dw_dz = (kp == km) ? 0. : (sample_velocity(w_values, i, j, kp) - sample_velocity(w_values, i, j, km)) / dz;

                    const double s00 = du_dx;
                    const double s11 = dv_dy;
                    const double s22 = dw_dz;
                    const double s01 = 0.5 * (du_dy + dv_dx);
                    const double s02 = 0.5 * (du_dz + dw_dx);
                    const double s12 = 0.5 * (dv_dz + dw_dy);
                    const double w01 = 0.5 * (du_dy - dv_dx);
                    const double w02 = 0.5 * (du_dz - dw_dx);
                    const double w12 = 0.5 * (dv_dz - dw_dy);
                    const double s_norm_sq = s00*s00 + s11*s11 + s22*s22
                            + 2.0 * (s01*s01 + s02*s02 + s12*s12);
                    const double w_norm_sq = 2.0 * (w01*w01 + w02*w02 + w12*w12);
                    const float q_value = static_cast<float>(w_norm_sq - s_norm_sq);

                    const openvdb::Coord coord(i, j, k);
                    density_accessor.setValueOn(coord, density);
                    q_accessor.setValueOn(coord, q_value);
                    ++active_voxels;
                }

        density_grid->insertMeta("microhh_active_voxels", openvdb::Int64Metadata(static_cast<std::int64_t>(active_voxels)));
        q_grid->insertMeta("microhh_active_voxels", openvdb::Int64Metadata(static_cast<std::int64_t>(active_voxels)));
        density_grid->tree().prune();
        q_grid->tree().prune();

        std::ostringstream filename;
        filename << "cloud_density_"
            << std::setw(6) << std::setfill('0') << frame
            << ".vdb";

        openvdb::GridPtrVec grids;
        grids.push_back(density_grid);
        grids.push_back(q_grid);
        openvdb::io::File file((directory / filename.str()).string());
        file.write(grids);

        summary.active_voxels += active_voxels;
    }

    return summary;
}

Snapshot Dataset::snapshot(
        const std::string& scalar_name,
        const int time_index,
        const int stride,
        const bool include_velocity) const
{
    const bool combined_scalar = scalar_name == "ql+qi";
    const auto& scalar = combined_scalar ? impl_->scalars.at("ql") : impl_->scalars.at(scalar_name);
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
        const auto qi_values = impl_->scalars.at("qi").read_time_slice(static_cast<std::size_t>(snapshot.time_index));
        if (qi_values.size() != scalar_values.size())
            throw std::runtime_error("ql+qi requires ql and qi to have identical dimensions");
        for (std::size_t id=0; id<scalar_values.size(); ++id)
            scalar_values[id] += qi_values[id];
    }

    std::map<std::string, std::vector<float>> velocity_values;
    if (include_velocity)
        for (const auto& [name, field] : impl_->velocity)
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
        const auto& field = impl_->velocity.at(name);
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
}
