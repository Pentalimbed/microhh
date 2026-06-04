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

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "master.h"
#include "input.h"
#include "grid.h"
#include "soil_grid.h"
#include "fields.h"
#include "buffer.h"
#include "netcdf_interface.h"
#include "timeloop.h"
#include "fft.h"
#include "boundary.h"
#include "immersed_boundary.h"
#include "advec.h"
#include "diff.h"
#include "pres.h"
#include "force.h"
#include "particle_bin.h"
#include "thermo.h"
#include "radiation.h"
#include "microphys.h"
#include "decay.h"
#include "limiter.h"
#include "stats.h"
#include "budget.h"
#include "column.h"
#include "cross.h"
#include "dump.h"
#include "model_viz.h"
#include "source.h"
#include "aerosol.h"
#include "background_profs.h"

#ifdef USECUDA
#include <cuda_runtime_api.h>
#endif

namespace
{
    void process_command_line_options(Sim_mode& sim_mode, std::string& sim_name,
                                      int argc, char *argv[],
                                      Master& master)
    {
        if (argc <= 1 || std::string(argv[1]) != "run")
            throw std::runtime_error("Specify run mode: microhh_viz run [case]");

        sim_mode = Sim_mode::Run;

        if (argc > 2)
            sim_name = argv[2];
        else
            sim_name = "microhh";

        master.print_message("Simulation name: %s\n", sim_name.c_str());
        master.print_message("Simulation mode: run\n");
    }
}

// In the constructor all classes are initialized and their input is read.
template<typename TF>
Model_viz<TF>::Model_viz(Master& masterin, int argc, char *argv[]) :
    master(masterin)
{
    process_command_line_options(sim_mode, sim_name, argc, argv, master);

    input = std::make_shared<Input>(master, sim_name + ".ini");
    // Keep the visualizer read/compute/render only, even for cases that
    // normally have statistics, columns, cross sections, or dumps enabled.
    input->set_item("stats", "swstats", "", "false");
    input->flag_as_used("stats", "sampletime", "");
    input->flag_as_used("stats", "masklist", "");
    input->flag_as_used("stats", "swtendency", "");
    input->flag_as_used("stats", "whitelist", "");
    input->flag_as_used("stats", "blacklist", "");
    input->set_item("column", "swcolumn", "", "false");
    input->set_item("cross", "swcross", "", "false");
    input->set_item("dump", "swdump", "", "false");
    input_nc = std::make_shared<Netcdf_file>(master, sim_name + "_input.nc", Netcdf_mode::Read);

    try
    {
        grid      = std::make_shared<Grid<TF>>     (master, *input);
        soil_grid = std::make_shared<Soil_grid<TF>>(master, *grid, *input);
        fields    = std::make_shared<Fields<TF>>   (master, *grid, *soil_grid, *input);
        timeloop  = std::make_shared<Timeloop<TF>> (master, *grid, *soil_grid, *fields, *input, sim_mode);
        fft       = std::make_shared<FFT<TF>>      (master, *grid);

        boundary  = Boundary<TF> ::factory(master, *grid, *soil_grid, *fields, *input);

        advec     = Advec<TF>    ::factory(master, *grid, *fields, *input);
        diff      = Diff<TF>     ::factory(master, *grid, *fields, *boundary, *input);
        pres      = Pres<TF>     ::factory(master, *grid, *fields, *fft, *input);
        thermo    = Thermo<TF>   ::factory(master, *grid, *fields, *input, sim_mode);
        microphys = Microphys<TF>::factory(master, *grid, *fields, *input);
        radiation = Radiation<TF>::factory(master, *grid, *fields, *input);

        force     = std::make_shared<Force  <TF>>(master, *grid, *fields, *input);
        buffer    = std::make_shared<Buffer <TF>>(master, *grid, *fields, *input);
        decay     = std::make_shared<Decay  <TF>>(master, *grid, *fields, *input);
        limiter   = std::make_shared<Limiter<TF>>(master, *grid, *fields, *diff, *input);
        source    = std::make_shared<Source <TF>>(master, *grid, *fields, *input);
        aerosol   = std::make_shared<Aerosol<TF>>(master, *grid, *fields, *input);
        background= std::make_shared<Background<TF>>(master, *grid, *fields, *input);

        particle_bin = std::make_shared<Particle_bin<TF>>(master, *grid, *fields, *input);

        ib        = std::make_shared<Immersed_boundary<TF>>(master, *grid, *fields, *input);

        stats     = std::make_shared<Stats <TF>>(master, *grid, *soil_grid, *background, *fields, *advec, *diff, *input);
        column    = std::make_shared<Column<TF>>(master, *grid, *fields, *input);
        dump      = std::make_shared<Dump  <TF>>(master, *grid, *fields, *input);
        cross     = std::make_shared<Cross <TF>>(master, *grid, *soil_grid, *fields, *input);

        budget    = Budget<TF>::factory(master, *grid, *fields, *thermo, *diff, *advec, *force, *stats, *input);

        // Parse the statistics masks
        add_statistics_masks();
    }
    catch (std::exception& e)
    {
        // In case of a failing constructor, delete the class objects and rethrow.
        delete_objects();
        throw;
    }
}

// In this function all instances of objects are deleted and the memory is freed.
template<typename TF>
void Model_viz<TF>::delete_objects()
{
}

// In the destructor the deletion of all class instances is triggered.
template<typename TF>
Model_viz<TF>::~Model_viz()
{
    delete_objects();
    #pragma omp taskwait
}

// In the init stage all class individual settings are known and the dynamic arrays are allocated.
template<typename TF>
void Model_viz<TF>::init()
{
    master.init(*input);

    grid->init();
    soil_grid->init();
    fields->init(*input, *dump, *cross, sim_mode);

    fft->init();

    boundary->init(*input, *thermo, sim_mode);
    ib->init(*input, *cross);
    buffer->init();
    diff->init();
    pres->init();
    force->init();
    thermo->init();
    microphys->init();
    radiation->init(*timeloop);
    decay->init(*input);
    budget->init();
    source->init();
    aerosol->init();
    background->init(*input_nc);

    stats->init();
    column->init();
    cross->init();
    dump->init();
}

template<typename TF>
void Model_viz<TF>::load_or_save()
{
    // The visualizer only supports run mode and therefore always starts from
    // existing restart/input files created by the normal MicroHH init stage.
    load();

    // This marks the end of the entire initialization.
    // Print warnings for input variables that are unused.
    input->print_unused_items();

    // Free the memory taken by the input fields.
    input.reset();
}

// In these functions data necessary to start the model is loaded from disk.
template<typename TF>
void Model_viz<TF>::load()
{
    // First load the grid and time to make their information available.
    grid->load(*input, *input_nc);
    fft->load();
    timeloop->load(timeloop->get_iotime());

    soil_grid->create(*input_nc);

    // Initialize the statistics file to open the possiblity to add profiles in other routines
    stats->create(*timeloop, sim_name);
    column->create(*input, *timeloop, sim_name);

    // Load the fields, and create the field statistics
    fields->load(timeloop->get_iotime());
    fields->load_rhoref();

    fields->create_stats(*stats);
    fields->create_column(*column);

    grid->create_stats(*stats);

    thermo->create(*input, *input_nc, *stats, *column, *cross, *dump, *timeloop);
    thermo->load(timeloop->get_iotime());

    boundary->load(timeloop->get_iotime(), *thermo);
    boundary->create(*input, *input_nc, *stats, *column, *cross, *timeloop);
    boundary->set_values();

    ib->create();
    buffer->create(*input, *input_nc, *stats);
    force->create(*input, *input_nc, *stats);
    source->create(*input, *input_nc);
    particle_bin->create(*timeloop);
    aerosol->create(*input, *input_nc, *stats);
    background->create(*input, *input_nc, *stats);

    microphys->create(*input, *input_nc, *stats, *cross, *dump, *column);

    // Radiation needs to be created after thermo as it needs base profiles.
    radiation->create(*input, *input_nc, *thermo, *stats, *column, *cross, *dump);
    decay->create(*input, *stats);
    limiter->create(*stats);

    // Cross and dump both need to be called at/near the
    // end of the create phase, as other classes register which
    // variables are legal as a cross/dump.
    cross->create();
    dump->create();

    pres->set_values();
    pres->create(*stats);
    advec->create(*stats);
    diff->create(*stats, false);

    thermo->create_stats(*stats);
    budget->create(*stats);

    restart_iotime = timeloop->get_iotime();
}

template<typename TF>
std::vector<std::string> Model_viz<TF>::scalar_names() const
{
    std::vector<std::string> names;
    names.reserve(fields->a.size());

    for (const auto& field : fields->a)
    {
        if (field.first == "u" || field.first == "v" || field.first == "w")
            continue;

        names.push_back(field.first);
    }

    return names;
}

template<typename TF>
bool Model_viz<TF>::has_velocity() const
{
    return fields->mp.count("u") && fields->mp.count("v") && fields->mp.count("w");
}

template<typename TF>
Viz_snapshot<TF> Model_viz<TF>::snapshot(const std::string& scalar_name, const int stride_in)
{
    #ifdef USECUDA
    if (!cpu_up_to_date)
    {
        #pragma omp taskwait
        fields   ->backward_device();
        boundary ->backward_device(*thermo);
        thermo   ->backward_device();
        microphys->backward_device();
        cpu_up_to_date = true;
    }
    #endif

    const int stride = std::max(1, stride_in);
    const auto scalar_it = fields->a.find(scalar_name);
    if (scalar_it == fields->a.end())
        throw std::runtime_error("Unknown scalar field \"" + scalar_name + "\"");

    const bool include_velocity = has_velocity();
    const auto& gd = grid->get_grid_data();

    Viz_snapshot<TF> out;
    out.nx = (gd.imax + stride - 1) / stride;
    out.ny = (gd.jmax + stride - 1) / stride;
    out.nz = (gd.kmax + stride - 1) / stride;
    out.time = timeloop->get_time();
    out.dt = timeloop->get_dt();
    out.iteration = timeloop->get_iteration();

    const int npoints = out.nx * out.ny * out.nz;
    out.points.reserve(3*npoints);
    out.scalars.reserve(npoints);
    if (include_velocity)
        out.vectors.reserve(3*npoints);

    out.scalar_min = std::numeric_limits<float>::max();
    out.scalar_max = std::numeric_limits<float>::lowest();
    out.vector_min = std::numeric_limits<float>::max();
    out.vector_max = std::numeric_limits<float>::lowest();

    const auto& scalar = scalar_it->second->fld;
    const auto& u = include_velocity ? fields->mp.at("u")->fld : scalar;
    const auto& v = include_velocity ? fields->mp.at("v")->fld : scalar;
    const auto& w = include_velocity ? fields->mp.at("w")->fld : scalar;

    for (int k=gd.kstart; k<gd.kend; k += stride)
        for (int j=gd.jstart; j<gd.jend; j += stride)
            for (int i=gd.istart; i<gd.iend; i += stride)
            {
                const int ijk = i + j*gd.icells + k*gd.ijcells;
                const float value = static_cast<float>(scalar[ijk]);

                out.points.push_back(static_cast<float>(gd.x[i]));
                out.points.push_back(static_cast<float>(gd.y[j]));
                out.points.push_back(static_cast<float>(gd.z[k]));

                out.scalars.push_back(value);
                out.scalar_min = std::min(out.scalar_min, value);
                out.scalar_max = std::max(out.scalar_max, value);

                if (include_velocity)
                {
                    const float uu = static_cast<float>(u[ijk] + gd.utrans);
                    const float vv = static_cast<float>(v[ijk] + gd.vtrans);
                    const float ww = static_cast<float>(w[ijk]);
                    const float magnitude = std::sqrt(uu*uu + vv*vv + ww*ww);

                    out.vectors.push_back(uu);
                    out.vectors.push_back(vv);
                    out.vectors.push_back(ww);
                    out.vector_min = std::min(out.vector_min, magnitude);
                    out.vector_max = std::max(out.vector_max, magnitude);
                }
            }

    if (out.scalars.empty())
    {
        out.scalar_min = 0.f;
        out.scalar_max = 0.f;
    }
    if (out.vectors.empty())
    {
        out.vector_min = 0.f;
        out.vector_max = 0.f;
    }

    return out;
}

template<typename TF>
void Model_viz<TF>::restart()
{
    #ifdef USECUDA
    if (gpu_prepared)
        clear_gpu();
    #endif

    master.print_message("Reloading visualizer restart at iotime %07d\n", restart_iotime);

    timeloop->load(restart_iotime);
    fields  ->load(restart_iotime);
    thermo  ->load(restart_iotime);
    boundary->load(restart_iotime, *thermo);
    boundary->set_values();
    pres->set_values();
    fields->reset_tendencies();

    #ifdef USECUDA
    if (gpu_prepared)
        prepare_gpu();
    cpu_up_to_date = false;
    #endif
}

template<typename TF>
double Model_viz<TF>::time() const
{
    return timeloop->get_time();
}

template<typename TF>
double Model_viz<TF>::dt() const
{
    return timeloop->get_dt();
}

template<typename TF>
int Model_viz<TF>::iteration() const
{
    return timeloop->get_iteration();
}

template<typename TF>
void Model_viz<TF>::prepare_run()
{
    #ifdef USECUDA
    prepare_gpu();
    gpu_prepared = true;
    #endif

    master.print_message("Starting visualizer time integration\n");

    #ifdef USECUDA
        #ifdef _OPENMP
        omp_set_nested(1);
        master.print_message("Running with %i OpenMP threads\n", omp_get_max_threads());
        #endif
    #else
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
    #endif
}

template<typename TF>
void Model_viz<TF>::finalize_run()
{
    #ifdef USECUDA
    // At the end of the run, copy the data back from the GPU.
    fields  ->backward_device();
    boundary->backward_device(*thermo);
    thermo  ->backward_device();

    clear_gpu();
    gpu_prepared = false;
    #endif
}

template<typename TF>
void Model_viz<TF>::advance_one_step()
{
    do
        advance_one_substep();
    while (timeloop->in_substep());
}

template<typename TF>
void Model_viz<TF>::advance_one_substep()
{
    // Update the time dependent parameters.
    grid      ->update_time_dependent(*timeloop);
    boundary  ->update_time_dependent(*timeloop);
    thermo    ->update_time_dependent(*timeloop);
    force     ->update_time_dependent(*timeloop);
    radiation ->update_time_dependent(*timeloop);
    aerosol   ->update_time_dependent(*timeloop);
    background->update_time_dependent(*timeloop);

    // Set the cyclic BCs of the prognostic 3D fields.
    boundary->set_prognostic_cyclic_bcs();
    boundary->set_prognostic_outflow_bcs();
    boundary->set_ghost_cells();

    // Calculate the field means, in case needed.
    fields->exec();

    // Get the viscosity to be used in diffusion.
    diff->exec_viscosity(*stats, *thermo);

    // Determine the time step.
    set_time_step();

    // File output is disabled in the visualizer, including status, statistics,
    // columns, cross sections, dumps, and restart writes.
    stats->set_tendency(false);

    // Calculate the thermodynamics and the buoyancy tendency.
    thermo->exec(timeloop->get_sub_time_step(), *stats);

    // Calculate the microphysics.
    microphys->exec(*thermo, timeloop->get_dt(), *stats);

    // Calculate the radiation fluxes and the related heating rate.
    radiation->exec(*thermo, timeloop->get_time(), *timeloop, *stats, *aerosol, *background, *microphys);

    // Calculate Monin-Obukhov parameters (L, u*), and calculate
    // surface fluxes, gradients, ...
    boundary->exec(*thermo, *radiation, *microphys, *timeloop);
    boundary->set_ghost_cells();

    // Set the immersed boundary conditions for scalars.
    ib->exec_scalars();

    // Update the outflow boundary conditions in case IB is used.
    if (ib->get_switch() != IB_type::Disabled)
        boundary->set_prognostic_outflow_bcs();

    // Calculate the advection tendency.
    boundary->set_ghost_cells_w(Boundary_w_type::Conservation_type);
    advec->exec(*stats);
    boundary->set_ghost_cells_w(Boundary_w_type::Normal_type);

    // Calculate the diffusion tendency.
    diff->exec(*stats);

    // Calculate the tendency due to damping in the buffer layer.
    buffer->exec(*stats);

    // Apply the scalar decay.
    decay->exec(timeloop->get_sub_time_step(), *stats);

    // Add point and line sources of scalars.
    source->exec(*timeloop);

    // Gravitational settling of binned dust types.
    particle_bin->exec(*stats);

    // Apply the large scale forcings. Keep this one always right before the pressure.
    force->exec(timeloop->get_sub_time_step(), *thermo, *stats);

    // Set the immersed boundary conditions.
    ib->exec_momentum();

    // Solve the Poisson equation for pressure.
    boundary->set_ghost_cells_w(Boundary_w_type::Conservation_type);
    pres->exec(timeloop->get_sub_time_step(), *stats);
    boundary->set_ghost_cells_w(Boundary_w_type::Normal_type);

    // Apply the limiter as the last tendency.
    limiter->exec(timeloop->get_sub_time_step(), *stats);

    // Integrate in time and advance by one RK substep. A full GUI "step"
    // repeats this until Timeloop returns to substep 0.
    timeloop->exec();
    timeloop->step_time();

    #ifdef USECUDA
    cpu_up_to_date = false;
    #endif
}

#ifdef USECUDA
template<typename TF>
void Model_viz<TF>::prepare_gpu()
{
    // Load all the necessary data to the GPU.
    master.print_message("Preparing the GPU\n");
    grid     ->prepare_device();
    soil_grid->prepare_device();
    fields   ->prepare_device();
    buffer   ->prepare_device();
    thermo   ->prepare_device();
    boundary ->prepare_device(*thermo);
    diff     ->prepare_device(*boundary);
    force    ->prepare_device();
    ib       ->prepare_device();
    microphys->prepare_device();
    radiation->prepare_device();
    column   ->prepare_device();
    aerosol  ->prepare_device();
    // Prepare pressure last, for memory check
    pres     ->prepare_device();
}

template<typename TF>
void Model_viz<TF>::clear_gpu()
{
    master.print_message("Clearing the GPU\n");
    grid     ->clear_device();
    soil_grid->clear_device();
    fields   ->clear_device();
    thermo   ->clear_device();
    boundary ->clear_device(*thermo);
    diff     ->clear_device();
    force    ->clear_device();
    ib       ->clear_device();
    microphys->clear_device();
    radiation->clear_device();
    column   ->clear_device();
    aerosol  ->clear_device();

    // Clear pressure last, for memory check
    pres     ->clear_device();
}
#endif

template<typename TF>
void Model_viz<TF>::set_time_step()
{
    // Only set the time step if the model is not in a substep.
    if (timeloop->in_substep())
        return;

    // Retrieve the maximum allowed time step per class.
    timeloop->set_time_step_limit();
    timeloop->set_time_step_limit(advec        ->get_time_limit(timeloop->get_idt(), timeloop->get_dt()));
    timeloop->set_time_step_limit(diff         ->get_time_limit(timeloop->get_idt(), timeloop->get_dt()));
    timeloop->set_time_step_limit(thermo       ->get_time_limit(timeloop->get_idt(), timeloop->get_dt()));
    timeloop->set_time_step_limit(microphys    ->get_time_limit(timeloop->get_idt(), timeloop->get_dt()));
    timeloop->set_time_step_limit(radiation    ->get_time_limit(timeloop->get_itime()));
    timeloop->set_time_step_limit(particle_bin->get_time_limit());

    // Set the time step.
    timeloop->set_time_step();
}

// Add all masks
template<typename TF>
void Model_viz<TF>::add_statistics_masks()
{
    const std::vector<std::string>& mask_list = stats->get_mask_list();

    // Check whether the mask can be retrieved from any of the mask-providing classes
    for (auto& mask_name : mask_list)
    {
        if (mask_name == "default")
            stats->add_mask(mask_name);
        else if (fields->has_mask(mask_name))
            stats->add_mask(mask_name);
        else if (thermo->has_mask(mask_name))
            stats->add_mask(mask_name);
        else if (microphys->has_mask(mask_name))
            stats->add_mask(mask_name);
        else if (decay->has_mask(mask_name))
            stats->add_mask(mask_name);
        else if (ib->has_mask(mask_name))
            stats->add_mask(mask_name);
        else
        {
            std::string error_message = "Can not calculate mask for \"" + mask_name + "\"";
            throw std::runtime_error(error_message);
        }
    }
}

#ifdef FLOAT_SINGLE
template class Model_viz<float>;
#else
template class Model_viz<double>;
#endif
