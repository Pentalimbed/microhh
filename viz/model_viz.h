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

#ifndef MODEL_VIZ_H
#define MODEL_VIZ_H

#include <string>
#include <memory>
#include <vector>

class Master;
class Input;
class Data_block;
class Netcdf_file;

template<typename> class Grid;
template<typename> class Soil_grid;
template<typename> class Fields;

template<typename> class Timeloop;
template<typename> class FFT;
template<typename> class Boundary;
template<typename> class Immersed_boundary;
template<typename> class Buffer;
template<typename> class Advec;
template<typename> class Diff;
template<typename> class Pres;
template<typename> class Force;
template<typename> class Aerosol;
template<typename> class Background;
template<typename> class Particle_bin;
template<typename> class Thermo;
template<typename> class Microphys;
template<typename> class Radiation;
template<typename> class Land_surface;

template<typename> class Decay;
template<typename> class Limiter;
template<typename> class Source;

template<typename> class Stats;
template<typename> class Budget;
template<typename> class Column;
template<typename> class Cross;
template<typename> class Dump;

enum class Sim_mode;

template<typename TF>
struct Viz_snapshot
{
    int nx = 0;
    int ny = 0;
    int nz = 0;

    double time = 0.;
    double dt = 0.;
    int iteration = 0;

    std::vector<float> points;
    std::vector<float> scalars;
    std::vector<float> vectors;

    float scalar_min = 0.f;
    float scalar_max = 0.f;
    float vector_min = 0.f;
    float vector_max = 0.f;
};

template<typename TF>
class Model_viz
{
    public:
        Model_viz(Master&, int, char**);
        ~Model_viz();

        void init();
        void load_or_save();
        void prepare_run();
        void finalize_run();

        std::vector<std::string> scalar_names() const;
        bool has_velocity() const;
        Viz_snapshot<TF> snapshot(const std::string&, int);

        void advance_one_step();
        void restart();

        double time() const;
        double dt() const;
        int iteration() const;

    private:
        Master& master;

        std::shared_ptr<Input> input;
        std::shared_ptr<Data_block> profs;
        std::shared_ptr<Netcdf_file> input_nc;

        std::shared_ptr<Grid<TF>> grid;
        std::shared_ptr<Soil_grid<TF>> soil_grid;
        std::shared_ptr<Fields<TF>> fields;

        std::shared_ptr<Timeloop<TF>> timeloop;

        std::shared_ptr<FFT<TF>> fft;

        std::shared_ptr<Boundary<TF>> boundary;
        std::shared_ptr<Immersed_boundary<TF>> ib;
        std::shared_ptr<Buffer<TF>> buffer;
        std::shared_ptr<Advec<TF>> advec;
        std::shared_ptr<Diff<TF>> diff;
        std::shared_ptr<Pres<TF>> pres;
        std::shared_ptr<Force<TF>> force;
        std::shared_ptr<Aerosol<TF>> aerosol;
        std::shared_ptr<Background<TF>> background;
        std::shared_ptr<Thermo<TF>> thermo;
        std::shared_ptr<Microphys<TF>> microphys;
        std::shared_ptr<Radiation<TF>> radiation;
        std::shared_ptr<Land_surface<TF>> lsm;

        std::shared_ptr<Decay<TF>> decay;
        std::shared_ptr<Limiter<TF>> limiter;
        std::shared_ptr<Source<TF>> source;

        std::shared_ptr<Particle_bin<TF>> particle_bin;

        std::shared_ptr<Stats<TF>> stats;
        std::shared_ptr<Budget<TF>> budget;
        std::shared_ptr<Column<TF>> column;
        std::shared_ptr<Cross<TF>> cross;
        std::shared_ptr<Dump<TF>> dump;

        Sim_mode sim_mode;
        std::string sim_name;
        bool cpu_up_to_date = false;
        bool gpu_prepared = false;
        int restart_iotime = 0;

        void load();

        void delete_objects();
        void advance_one_substep();
        void set_time_step();

        void prepare_gpu();
        void clear_gpu();

        void add_statistics_masks();
};
#endif
