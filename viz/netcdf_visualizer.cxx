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
#include "netcdf_viewer.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvdb/openvdb.h>

namespace
{
    void print_usage()
    {
        std::cout
            << "Usage: microhh_nc_viz [directory]\n"
            << "       microhh_nc_viz --directory CASE_DIR\n"
            << "       microhh_nc_viz [--output-top M] [--vertical-cells N] [--inspect] INPUT...\n"
            << "       microhh_nc_viz ql.nc qi.nc qr.nc qs.nc qg.nc u.nc v.nc w.nc\n\n"
            << "The viewer reads NetCDF files produced by python/3d_to_nc.py.\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string> args;
        std::optional<double> output_top;
        std::optional<int> vertical_cells;
        bool inspect = false;
        for (int i=1; i<argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--inspect")
                inspect = true;
            else if (argument == "--output-top" && i+1 < argc)
                output_top = std::stod(argv[++i]);
            else if (argument == "--vertical-cells" && i+1 < argc)
                vertical_cells = std::stoi(argv[++i]);
            else
                args.push_back(argument);
        }

        if (std::find(args.begin(), args.end(), "-h") != args.end()
                || std::find(args.begin(), args.end(), "--help") != args.end())
        {
            print_usage();
            return 0;
        }

        const auto paths = microhh::viz::discover_paths(args);
        const auto source_directory = microhh::viz::common_parent_directory(paths);
        const auto case_settings = microhh::viz::read_case_settings(source_directory);
        openvdb::initialize();
        microhh::viz::Dataset dataset(paths);
        auto output_grid = dataset.default_output_grid();
        if (case_settings.has_domain_size)
            output_grid.top = case_settings.domain_size[2];
        if (output_top)
            output_grid.top = *output_top;
        if (vertical_cells)
            output_grid.vertical_cells = *vertical_cells;
        if (!(output_grid.top > 0.) || output_grid.vertical_cells <= 0)
            throw std::runtime_error("Output top and vertical cell count must be positive");

        if (inspect)
        {
            const auto& scalar_names = dataset.scalar_names();
            const auto combined = std::find(scalar_names.begin(), scalar_names.end(), "ql+qi");
            const auto& scalar_name = combined == scalar_names.end() ? scalar_names.front() : *combined;
            const auto snapshot = dataset.snapshot(scalar_name, 0, 1, false, output_grid);
            std::cout
                << "field=" << scalar_name << "\n"
                << "cells=" << snapshot.nx << " " << snapshot.ny << " " << snapshot.nz << "\n"
                << "origin_m=" << snapshot.domain_origin[0] << " " << snapshot.domain_origin[1] << " " << snapshot.domain_origin[2] << "\n"
                << "size_m=" << snapshot.domain_size[0] << " " << snapshot.domain_size[1] << " " << snapshot.domain_size[2] << "\n"
                << "cell_m=" << snapshot.cell_size[0] << " " << snapshot.cell_size[1] << " " << snapshot.cell_size[2] << "\n"
                << "extinction_m-1=" << snapshot.scalar_min << " " << snapshot.scalar_max << "\n";
            return 0;
        }

        microhh::viz::run_visualizer(
                dataset, case_settings, source_directory, output_grid);
    }
    catch (const std::exception& e)
    {
        std::cerr << "microhh_nc_viz: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
