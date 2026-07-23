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
#include <cmath>
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

    // Representative visible-light mass-extinction coefficients in m2/kg.
    constexpr float cloud_water_extinction = 144.7f;
    constexpr float cloud_ice_extinction = 65.f;
    constexpr float rain_extinction = 3.f;
    constexpr float snow_extinction = 15.f;
    constexpr float graupel_extinction = 8.f;

    float mass_extinction_coefficient(const std::string& name)
    {
        if (name == "ql") return cloud_water_extinction;
        if (name == "qi") return cloud_ice_extinction;
        if (name == "qr") return rain_extinction;
        if (name == "qs") return snow_extinction;
        if (name == "qg") return graupel_extinction;
        return 1.f;
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

    std::array<double, 2> inferred_axis_bounds(const std::vector<float>& axis)
    {
        if (axis.empty())
            return {{0., 1.}};
        if (axis.size() == 1)
            return {{static_cast<double>(axis.front()) - 0.5,
                     static_cast<double>(axis.front()) + 0.5}};

        return {{
            static_cast<double>(axis.front()) - 0.5*static_cast<double>(axis[1] - axis[0]),
            static_cast<double>(axis.back()) + 0.5*static_cast<double>(axis.back() - axis[axis.size()-2])}};
    }

    struct Cubic_axis_sample
    {
        std::array<int, 4> index = {{0, 0, 0, 0}};
        std::array<float, 4> weight = {{1.f, 0.f, 0.f, 0.f}};
        bool inside = true;
    };

    Cubic_axis_sample locate_cubic_axis_sample(
            const std::vector<float>& axis,
            const float value)
    {
        Cubic_axis_sample sample;
        if (axis.empty())
        {
            sample.inside = false;
            return sample;
        }
        if (axis.size() == 1)
            return sample;

        const auto bounds = inferred_axis_bounds(axis);
        const double bound_min = std::min(bounds[0], bounds[1]);
        const double bound_max = std::max(bounds[0], bounds[1]);
        sample.inside = value >= bound_min && value <= bound_max;

        const double equality_tolerance = std::max(1.e-6, (bound_max-bound_min)*1.e-7);
        const auto nearest = std::lower_bound(axis.begin(), axis.end(), value);
        if (axis.front() <= axis.back() && nearest != axis.end())
        {
            const int index = static_cast<int>(std::distance(axis.begin(), nearest));
            if (std::abs(static_cast<double>(*nearest)-value) <= equality_tolerance)
            {
                sample.index = {{index, index, index, index}};
                return sample;
            }
        }

        const bool ascending = axis.front() < axis.back();
        int hi = 1;
        if (ascending)
        {
            const auto it = std::upper_bound(axis.begin(), axis.end(), value);
            hi = std::clamp(static_cast<int>(std::distance(axis.begin(), it)), 1,
                    static_cast<int>(axis.size()) - 1);
        }
        else
        {
            const auto it = std::upper_bound(axis.begin(), axis.end(), value, std::greater<float>());
            hi = std::clamp(static_cast<int>(std::distance(axis.begin(), it)), 1,
                    static_cast<int>(axis.size()) - 1);
        }

        const int first = std::clamp(hi - 2, 0, std::max(0, static_cast<int>(axis.size()) - 4));
        for (int n=0; n<4; ++n)
            sample.index[static_cast<std::size_t>(n)] = std::min(first + n, static_cast<int>(axis.size()) - 1);

        if (axis.size() < 4)
        {
            const auto linear = locate_axis_sample(axis, value);
            sample.index = {{linear.lo, linear.hi, linear.hi, linear.hi}};
            sample.weight = {{1.f-linear.weight, linear.weight, 0.f, 0.f}};
            return sample;
        }

        for (int n=0; n<4; ++n)
        {
            double weight = 1.;
            const double xn = axis[static_cast<std::size_t>(sample.index[static_cast<std::size_t>(n)])];
            for (int m=0; m<4; ++m)
            {
                if (m == n)
                    continue;
                const double xm = axis[static_cast<std::size_t>(sample.index[static_cast<std::size_t>(m)])];
                const double denominator = xn - xm;
                if (denominator == 0.)
                {
                    weight = 0.;
                    break;
                }
                weight *= (static_cast<double>(value) - xm) / denominator;
            }
            sample.weight[static_cast<std::size_t>(n)] = static_cast<float>(weight);
        }
        return sample;
    }

    float sample_component_cubic(
            const std::vector<float>& values,
            const int nx,
            const int ny,
            const Cubic_axis_sample& x,
            const Cubic_axis_sample& y,
            const Cubic_axis_sample& z,
            const bool zero_outside)
    {
        if (zero_outside && (!x.inside || !y.inside || !z.inside))
            return 0.f;

        double result = 0.;
        for (int kk=0; kk<4; ++kk)
        {
            const double wz = z.weight[static_cast<std::size_t>(kk)];
            if (wz == 0.)
                continue;
            for (int jj=0; jj<4; ++jj)
            {
                const double wy = y.weight[static_cast<std::size_t>(jj)];
                if (wy == 0.)
                    continue;
                for (int ii=0; ii<4; ++ii)
                {
                    const double wx = x.weight[static_cast<std::size_t>(ii)];
                    if (wx == 0.)
                        continue;
                    result += wx * wy * wz
                            * values[point_id(
                                    nx, ny,
                                    x.index[static_cast<std::size_t>(ii)],
                                    y.index[static_cast<std::size_t>(jj)],
                                    z.index[static_cast<std::size_t>(kk)])];
                }
            }
        }
        return static_cast<float>(result);
    }

    std::vector<float> uniform_cell_centers(
            const double lower,
            const double upper,
            const int count)
    {
        if (!(upper > lower) || count <= 0)
            throw std::runtime_error("Output grid bounds and cell counts must be positive");

        const double spacing = (upper - lower) / static_cast<double>(count);
        std::vector<float> axis(static_cast<std::size_t>(count));
        for (int i=0; i<count; ++i)
            axis[static_cast<std::size_t>(i)] = static_cast<float>(lower + (static_cast<double>(i) + 0.5)*spacing);
        return axis;
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

    openvdb::FloatGrid::Ptr make_float_grid(
            const std::string& name,
            const std::string& source,
            const openvdb::math::Transform::Ptr& transform,
            const double time,
            const std::size_t frame,
            const bool resampled_to_uniform,
            const openvdb::GridClass grid_class)
    {
        auto grid = openvdb::FloatGrid::create(0.f);
        grid->setName(name);
        grid->setGridClass(grid_class);
        grid->setTransform(transform->copy());
        grid->insertMeta("microhh_source", openvdb::StringMetadata(source));
        grid->insertMeta("microhh_time", openvdb::DoubleMetadata(time));
        grid->insertMeta("microhh_time_index", openvdb::Int32Metadata(static_cast<std::int32_t>(frame)));
        grid->insertMeta("microhh_resampled_to_uniform", openvdb::StringMetadata(resampled_to_uniform ? "true" : "false"));
        return grid;
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

    const std::array<std::string, 5> hydrometeors = {{"ql", "qi", "qr", "qs", "qg"}};
    impl_->scalar_names.erase(
            std::remove_if(
                impl_->scalar_names.begin(), impl_->scalar_names.end(),
                [&](const std::string& name)
                {
                    return std::find(hydrometeors.begin(), hydrometeors.end(), name) == hydrometeors.end();
                }),
            impl_->scalar_names.end());
    std::sort(
            impl_->scalar_names.begin(), impl_->scalar_names.end(),
            [&](const std::string& a, const std::string& b)
            {
                return std::find(hydrometeors.begin(), hydrometeors.end(), a)
                        < std::find(hydrometeors.begin(), hydrometeors.end(), b);
            });
    if (impl_->scalars.count("ql") && impl_->scalars.count("qi"))
        impl_->scalar_names.push_back("ql+qi");
    if (impl_->scalar_names.empty())
        throw std::runtime_error("No supported hydrometeor fields were loaded. Expected ql, qi, qr, qs, or qg.");
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

bool Dataset::has_cloud_velocity_fields() const
{
    return impl_->scalars.count("ql")
            && impl_->scalars.count("qi")
            && impl_->velocity.count("u")
            && impl_->velocity.count("v")
            && impl_->velocity.count("w");
}

Output_grid Dataset::default_output_grid() const
{
    if (impl_->scalars.empty())
        return {};

    const auto& field = impl_->scalars.begin()->second;
    const auto bounds = inferred_axis_bounds(field.z);
    return {std::max(bounds[0], bounds[1]), static_cast<int>(field.nz())};
}

int Dataset::time_count(const std::string& scalar_name) const
{
    if (scalar_name == "ql+qi")
        return static_cast<int>(std::min(impl_->scalars.at("ql").nt(), impl_->scalars.at("qi").nt()));
    return static_cast<int>(impl_->scalars.at(scalar_name).nt());
}

Vdb_export_summary Dataset::export_cloud_velocity_vdb_sequence(
        const fs::path& directory,
        const Output_grid& output_grid) const
{
    if (!has_cloud_velocity_fields())
        throw std::runtime_error("VDB export requires ql.nc, qi.nc, u.nc, v.nc, and w.nc");

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

    struct Velocity_export_sampler
    {
        const Field_file* field = nullptr;
        std::vector<Cubic_axis_sample> x;
        std::vector<Cubic_axis_sample> y;
        std::vector<Cubic_axis_sample> z;
    };

    fs::create_directories(directory);

    const int nx = static_cast<int>(ql.nx());
    const int ny = static_cast<int>(ql.ny());
    const int nz = output_grid.vertical_cells;
    const auto x_bounds = inferred_axis_bounds(ql.x);
    const auto y_bounds = inferred_axis_bounds(ql.y);
    const double x_min = std::min(x_bounds[0], x_bounds[1]);
    const double x_max = std::max(x_bounds[0], x_bounds[1]);
    const double y_min = std::min(y_bounds[0], y_bounds[1]);
    const double y_max = std::max(y_bounds[0], y_bounds[1]);
    const double z_bottom = 0.;
    if (!(output_grid.top > z_bottom) || nz <= 0)
        throw std::runtime_error("Output top must exceed the input bottom and vertical cells must be positive");

    const auto export_x = uniform_cell_centers(x_min, x_max, nx);
    const auto export_y = uniform_cell_centers(y_min, y_max, ny);
    const auto export_z = uniform_cell_centers(z_bottom, output_grid.top, nz);
    std::vector<Cubic_axis_sample> x_samples;
    std::vector<Cubic_axis_sample> y_samples;
    std::vector<Cubic_axis_sample> z_samples;
    for (const float value : export_x) x_samples.push_back(locate_cubic_axis_sample(ql.x, value));
    for (const float value : export_y) y_samples.push_back(locate_cubic_axis_sample(ql.y, value));
    for (const float value : export_z) z_samples.push_back(locate_cubic_axis_sample(ql.z, value));

    std::map<std::string, Velocity_export_sampler> velocity_samplers;
    for (const auto& name : {"u", "v", "w"})
    {
        const auto& field = impl_->velocity.at(name);
        velocity_samplers.emplace(
                name,
                Velocity_export_sampler{
                    &field,
                    {}, {}, {}});
        auto& sampler = velocity_samplers.at(name);
        for (const float value : export_x) sampler.x.push_back(locate_cubic_axis_sample(field.x, value));
        for (const float value : export_y) sampler.y.push_back(locate_cubic_axis_sample(field.y, value));
        for (const float value : export_z) sampler.z.push_back(locate_cubic_axis_sample(field.z, value));
    }

    Vdb_export_summary summary;
    summary.frames = frame_count;
    summary.resampled = true;

    for (std::size_t frame=0; frame<frame_count; ++frame)
    {
        const auto ql_values = ql.read_time_slice(frame);
        const auto qi_values = qi.read_time_slice(frame);
        if (qi_values.size() != ql_values.size())
            throw std::runtime_error("ql and qi frame sizes do not match");

        std::map<std::string, std::vector<float>> velocity_values;
        for (const auto& [name, sampler] : velocity_samplers)
        {
            const auto velocity_frame = std::min<std::size_t>(frame, sampler.field->nt() - 1);
            auto values = sampler.field->read_time_slice(velocity_frame);
            const std::size_t expected_size = sampler.field->nx() * sampler.field->ny() * sampler.field->nz();
            if (values.size() != expected_size)
                throw std::runtime_error(name + " frame size does not match its dimensions");
            velocity_values.emplace(name, std::move(values));
        }

        auto transform = openvdb::math::Transform::createLinearTransform(1.);
        const std::array<double, 3> domain_size = {{x_max-x_min, y_max-y_min, output_grid.top-z_bottom}};
        const std::array<double, 3> spacing = {{domain_size[0]/nx, domain_size[1]/ny, domain_size[2]/nz}};
        transform->postScale(openvdb::math::Vec3d(
                    spacing[0], spacing[1], spacing[2]));
        transform->postTranslate(openvdb::math::Vec3d(
                    -0.5*domain_size[0] + 0.5*spacing[0],
                    -0.5*domain_size[1] + 0.5*spacing[1],
                    z_bottom + 0.5*spacing[2]));

        auto ql_grid = make_float_grid(
                "ql", "ql", transform, ql.time_value(frame), frame,
                true, openvdb::GRID_FOG_VOLUME);
        auto qi_grid = make_float_grid(
                "qi", "qi", transform, ql.time_value(frame), frame,
                true, openvdb::GRID_FOG_VOLUME);
        auto u_grid = make_float_grid(
                "u", "u", transform, ql.time_value(frame), frame,
                true, openvdb::GRID_UNKNOWN);
        auto v_grid = make_float_grid(
                "v", "v", transform, ql.time_value(frame), frame,
                true, openvdb::GRID_UNKNOWN);
        auto w_grid = make_float_grid(
                "w", "w", transform, ql.time_value(frame), frame,
                true, openvdb::GRID_UNKNOWN);

        const openvdb::Vec3d domain_origin(
                -0.5*domain_size[0], -0.5*domain_size[1], z_bottom);
        const openvdb::Vec3d output_size(domain_size[0], domain_size[1], domain_size[2]);
        const openvdb::Vec3d voxel_size(spacing[0], spacing[1], spacing[2]);
        for (auto& grid : {ql_grid, qi_grid, u_grid, v_grid, w_grid})
        {
            grid->insertMeta("microhh_domain_origin_m", openvdb::Vec3DMetadata(domain_origin));
            grid->insertMeta("microhh_domain_size_m", openvdb::Vec3DMetadata(output_size));
            grid->insertMeta("microhh_voxel_size_m", openvdb::Vec3DMetadata(voxel_size));
            grid->insertMeta("microhh_domain_placement", openvdb::StringMetadata(
                        "origin is centered laterally on the bottom domain face; transform maps ijk to cell centers"));
        }

        auto ql_accessor = ql_grid->getAccessor();
        auto qi_accessor = qi_grid->getAccessor();
        auto u_accessor = u_grid->getAccessor();
        auto v_accessor = v_grid->getAccessor();
        auto w_accessor = w_grid->getAccessor();
        std::size_t active_voxels = 0;

        const auto sample_scalar = [&](const std::vector<float>& values, const int i, const int j, const int k)
        {
            return std::max(0.f, sample_component_cubic(
                    values, nx, ny,
                    x_samples[static_cast<std::size_t>(i)],
                    y_samples[static_cast<std::size_t>(j)],
                    z_samples[static_cast<std::size_t>(k)], true));
        };

        const auto sample_velocity = [&](const std::string& name, const int i, const int j, const int k)
        {
            const auto& sampler = velocity_samplers.at(name);
            return sample_component_cubic(
                    velocity_values.at(name),
                    static_cast<int>(sampler.field->nx()),
                    static_cast<int>(sampler.field->ny()),
                    sampler.x[static_cast<std::size_t>(i)],
                    sampler.y[static_cast<std::size_t>(j)],
                    sampler.z[static_cast<std::size_t>(k)], true);
        };

        for (int k=0; k<nz; ++k)
            for (int j=0; j<ny; ++j)
                for (int i=0; i<nx; ++i)
                {
                    const float ql_value = sample_scalar(ql_values, i, j, k);
                    const float qi_value = sample_scalar(qi_values, i, j, k);
                    const openvdb::Coord coord(i, j, k);

                    if (ql_value != 0.f)
                        ql_accessor.setValueOn(coord, ql_value);
                    if (qi_value != 0.f)
                        qi_accessor.setValueOn(coord, qi_value);

                    if (ql_value == 0.f && qi_value == 0.f)
                        continue;

                    u_accessor.setValueOn(coord, sample_velocity("u", i, j, k));
                    v_accessor.setValueOn(coord, sample_velocity("v", i, j, k));
                    w_accessor.setValueOn(coord, sample_velocity("w", i, j, k));
                    ++active_voxels;
                }

        for (auto& grid : {ql_grid, qi_grid, u_grid, v_grid, w_grid})
        {
            grid->insertMeta("microhh_active_cloud_voxels", openvdb::Int64Metadata(static_cast<std::int64_t>(active_voxels)));
            grid->tree().prune();
        }

        std::ostringstream filename;
        filename << "cloud_fields_"
            << std::setw(6) << std::setfill('0') << frame
            << ".vdb";

        openvdb::GridPtrVec grids;
        grids.push_back(ql_grid);
        grids.push_back(qi_grid);
        grids.push_back(u_grid);
        grids.push_back(v_grid);
        grids.push_back(w_grid);
        openvdb::io::File file((directory / filename.str()).string());
        file.write(grids);

        summary.active_voxels += active_voxels;
    }

    return summary;
}

Snapshot Dataset::snapshot(
        const std::string& scalar_name,
        const int time_index,
        const int horizontal_stride,
        const bool include_velocity,
        const Output_grid& output_grid) const
{
    const bool combined_scalar = scalar_name == "ql+qi";
    const auto& scalar = combined_scalar ? impl_->scalars.at("ql") : impl_->scalars.at(scalar_name);
    const int source_nx = static_cast<int>(scalar.nx());
    const int source_ny = static_cast<int>(scalar.ny());
    const int s = std::max(1, horizontal_stride);
    const auto x_bounds = inferred_axis_bounds(scalar.x);
    const auto y_bounds = inferred_axis_bounds(scalar.y);
    const double x_min = std::min(x_bounds[0], x_bounds[1]);
    const double x_max = std::max(x_bounds[0], x_bounds[1]);
    const double y_min = std::min(y_bounds[0], y_bounds[1]);
    const double y_max = std::max(y_bounds[0], y_bounds[1]);
    const double z_bottom = 0.;
    const double z_top = output_grid.top;
    if (!(z_top > z_bottom) || output_grid.vertical_cells <= 0)
        throw std::runtime_error("Output top must exceed the input bottom and vertical cells must be positive");

    Snapshot snapshot;
    snapshot.nx = (source_nx + s - 1) / s;
    snapshot.ny = (source_ny + s - 1) / s;
    snapshot.nz = output_grid.vertical_cells;
    const auto source_x = uniform_cell_centers(x_min, x_max, snapshot.nx);
    const auto source_y = uniform_cell_centers(y_min, y_max, snapshot.ny);
    snapshot.z = uniform_cell_centers(z_bottom, z_top, snapshot.nz);
    snapshot.domain_size = {{x_max-x_min, y_max-y_min, z_top-z_bottom}};
    snapshot.domain_origin = {{-0.5*snapshot.domain_size[0], -0.5*snapshot.domain_size[1], z_bottom}};
    snapshot.cell_size = {{
        snapshot.domain_size[0] / snapshot.nx,
        snapshot.domain_size[1] / snapshot.ny,
        snapshot.domain_size[2] / snapshot.nz}};
    snapshot.x = uniform_cell_centers(
            snapshot.domain_origin[0], snapshot.domain_origin[0] + snapshot.domain_size[0], snapshot.nx);
    snapshot.y = uniform_cell_centers(
            snapshot.domain_origin[1], snapshot.domain_origin[1] + snapshot.domain_size[1], snapshot.ny);
    snapshot.time_index = std::clamp(time_index, 0, static_cast<int>(scalar.nt()) - 1);
    snapshot.time = scalar.time_value(static_cast<std::size_t>(snapshot.time_index));

    auto scalar_values = scalar.read_time_slice(static_cast<std::size_t>(snapshot.time_index));
    std::vector<float> ice_values;
    if (combined_scalar)
    {
        const auto& ice = impl_->scalars.at("qi");
        if (scalar.nx() != ice.nx() || scalar.ny() != ice.ny() || scalar.nz() != ice.nz()
                || !axes_match_for_combination(scalar.x, ice.x)
                || !axes_match_for_combination(scalar.y, ice.y)
                || !axes_match_for_combination(scalar.z, ice.z))
            throw std::runtime_error("ql+qi requires matching dimensions and coordinate axes");
        ice_values = ice.read_time_slice(static_cast<std::size_t>(snapshot.time_index));
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

    std::vector<Cubic_axis_sample> scalar_x_samples;
    std::vector<Cubic_axis_sample> scalar_y_samples;
    std::vector<Cubic_axis_sample> scalar_z_samples;
    for (const float value : source_x)
        scalar_x_samples.push_back(locate_cubic_axis_sample(scalar.x, value));
    for (const float value : source_y)
        scalar_y_samples.push_back(locate_cubic_axis_sample(scalar.y, value));
    for (const float value : snapshot.z)
        scalar_z_samples.push_back(locate_cubic_axis_sample(scalar.z, value));

    struct Velocity_sampler
    {
        const Field_file* field = nullptr;
        const std::vector<float>* values = nullptr;
        std::vector<Cubic_axis_sample> x;
        std::vector<Cubic_axis_sample> y;
        std::vector<Cubic_axis_sample> z;
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
                    {}, {}, {}});
        auto& sampler = velocity_samplers.at(name);
        for (const float value : source_x)
            sampler.x.push_back(locate_cubic_axis_sample(field.x, value));
        for (const float value : source_y)
            sampler.y.push_back(locate_cubic_axis_sample(field.y, value));
        for (const float value : snapshot.z)
            sampler.z.push_back(locate_cubic_axis_sample(field.z, value));
    }

    for (int k=0; k<snapshot.nz; ++k)
        for (int j=0; j<snapshot.ny; ++j)
            for (int i=0; i<snapshot.nx; ++i)
            {
                const auto id = point_id(snapshot.nx, snapshot.ny, i, j, k);

                snapshot.points[3*id] = snapshot.x[static_cast<std::size_t>(i)];
                snapshot.points[3*id + 1] = snapshot.y[static_cast<std::size_t>(j)];
                snapshot.points[3*id + 2] = snapshot.z[static_cast<std::size_t>(k)];
                float extinction = mass_extinction_coefficient(scalar.name) * sample_component_cubic(
                        scalar_values, source_nx, source_ny,
                        scalar_x_samples[static_cast<std::size_t>(i)],
                        scalar_y_samples[static_cast<std::size_t>(j)],
                        scalar_z_samples[static_cast<std::size_t>(k)], true);
                if (combined_scalar)
                    extinction += cloud_ice_extinction * sample_component_cubic(
                            ice_values, source_nx, source_ny,
                            scalar_x_samples[static_cast<std::size_t>(i)],
                            scalar_y_samples[static_cast<std::size_t>(j)],
                            scalar_z_samples[static_cast<std::size_t>(k)], true);
                snapshot.scalars[id] = extinction > 1.e-12f ? extinction : 0.f;
                update_minmax(snapshot.scalars[id], snapshot.scalar_min, snapshot.scalar_max);

                if (!snapshot.vectors.empty())
                {
                    float u = 0.f;
                    float v = 0.f;
                    float w = 0.f;

                    if (const auto it = velocity_samplers.find("u"); it != velocity_samplers.end())
                        u = sample_component_cubic(
                                *it->second.values,
                                static_cast<int>(it->second.field->nx()),
                                static_cast<int>(it->second.field->ny()),
                                it->second.x[static_cast<std::size_t>(i)],
                                it->second.y[static_cast<std::size_t>(j)],
                                it->second.z[static_cast<std::size_t>(k)], true);
                    if (const auto it = velocity_samplers.find("v"); it != velocity_samplers.end())
                        v = sample_component_cubic(
                                *it->second.values,
                                static_cast<int>(it->second.field->nx()),
                                static_cast<int>(it->second.field->ny()),
                                it->second.x[static_cast<std::size_t>(i)],
                                it->second.y[static_cast<std::size_t>(j)],
                                it->second.z[static_cast<std::size_t>(k)], true);
                    if (const auto it = velocity_samplers.find("w"); it != velocity_samplers.end())
                        w = sample_component_cubic(
                                *it->second.values,
                                static_cast<int>(it->second.field->nx()),
                                static_cast<int>(it->second.field->ny()),
                                it->second.x[static_cast<std::size_t>(i)],
                                it->second.y[static_cast<std::size_t>(j)],
                                it->second.z[static_cast<std::size_t>(k)], true);

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
        for (const char* name : {"ql.nc", "qi.nc", "qr.nc", "qs.nc", "qg.nc", "u.nc", "v.nc", "w.nc"})
        {
            auto path = directory / name;
            if (fs::exists(path))
                paths.push_back(path);
        }
    }

    if (paths.empty())
        throw std::runtime_error("No NetCDF files found. Pass a directory containing hydrometeor .nc files or explicit .nc paths.");

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
