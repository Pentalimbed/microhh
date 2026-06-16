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

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace microhh::viz
{
    namespace fs = std::filesystem;

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

    struct Vdb_export_summary
    {
        std::size_t frames = 0;
        std::size_t active_voxels = 0;
        bool resampled = false;
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

    struct Axis_sample
    {
        int lo = 0;
        int hi = 0;
        float weight = 0.f;
    };

    inline std::size_t point_id(const int nx, const int ny, const int i, const int j, const int k)
    {
        return static_cast<std::size_t>((k*ny + j)*nx + i);
    }

    inline Axis_sample locate_axis_sample(const std::vector<float>& axis, const float value)
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

    inline std::vector<Axis_sample> map_axis_samples(
            const std::vector<float>& target_axis,
            const std::vector<float>& source_axis)
    {
        std::vector<Axis_sample> samples;
        samples.reserve(target_axis.size());
        for (const auto value : target_axis)
            samples.push_back(locate_axis_sample(source_axis, value));
        return samples;
    }

    inline float lerp(const float a, const float b, const float weight)
    {
        return a + weight*(b - a);
    }

    inline float sample_component(
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

    class Dataset
    {
        public:
            explicit Dataset(const std::vector<fs::path>& paths);
            ~Dataset();

            Dataset(Dataset&&) noexcept;
            Dataset& operator=(Dataset&&) noexcept;

            Dataset(const Dataset&) = delete;
            Dataset& operator=(const Dataset&) = delete;

            const std::vector<std::string>& scalar_names() const;
            bool has_velocity() const;
            bool has_cloud_velocity_fields() const;
            int time_count(const std::string& scalar_name) const;
            Vdb_export_summary export_cloud_velocity_vdb_sequence(const fs::path& directory) const;
            Snapshot snapshot(
                    const std::string& scalar_name,
                    int time_index,
                    int stride,
                    bool include_velocity) const;

        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
    };

    Case_settings read_case_settings(const fs::path& directory);
    std::vector<fs::path> discover_paths(const std::vector<std::string>& args);
    fs::path common_parent_directory(const std::vector<fs::path>& paths);
}
