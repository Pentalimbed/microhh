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
            << "       microhh_nc_viz ql.nc qi.nc u.nc v.nc w.nc\n\n"
            << "The viewer reads NetCDF files produced by python/3d_to_nc.py.\n";
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

        const auto paths = microhh::viz::discover_paths(args);
        const auto source_directory = microhh::viz::common_parent_directory(paths);
        const auto case_settings = microhh::viz::read_case_settings(source_directory);
        openvdb::initialize();
        microhh::viz::Dataset dataset(paths);
        microhh::viz::run_visualizer(dataset, case_settings, source_directory);
    }
    catch (const std::exception& e)
    {
        std::cerr << "microhh_nc_viz: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
