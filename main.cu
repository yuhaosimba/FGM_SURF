#include "main.cuh"

#define SUBPACKAGE_HINT "SPONGE, for normal molecular dynamics simulations"

CONTROLLER controller;
MD_INFORMATION md_info;
MIDDLE_Langevin_INFORMATION middle_langevin;
Langevin_MD_INFORMATION langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
LISTED_FORCES listed_forces;
BOND bond;
ANGLE angle;
UREY_BRADLEY urey_bradley;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
CMAP cmap;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
SOLVENT_LENNARD_JONES solvent_lj;
Particle_Mesh_Ewald pme;
LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SIMPLE_CONSTRAIN simple_constrain;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
MC_BAROSTAT_INFORMATION mc_baro;
BERENDSEN_BAROSTAT_INFORMATION bd_baro;
ANDERSEN_BAROSTAT_INFORMATION ad_baro;
LJ_SOFT_CORE lj_soft;
PAIRWISE_FORCE pairwise_force;
BOND_SOFT bond_soft;
SITS_INFORMATION sits;
DIHEDRAL sits_dihedral;
RESTRAIN_CV restrain_cv;
STEER_CV steer_cv;
COLLECTIVE_VARIABLE_CONTROLLER cv_controller;
META1D meta;
HARD_WALL hard_wall;
SOFT_WALLS soft_walls;
SPONGE_PLUGIN plugin;
FGM_SURF fgm_surf;



int main(int argc, char *argv[])
{
    Main_Initial(argc, argv);
    for (md_info.sys.steps = 1; md_info.sys.steps <= md_info.sys.step_limit; md_info.sys.steps++)
    {
        Main_Calculate_Force();
        
       
        // DEBUG FGM procdure
        FGM_CalC_Force_Only_with_Exclude(fgm_surf, md_info.crd, md_info.d_charge,md_info.nb.h_excluded_list_start, md_info.nb.h_excluded_list, md_info.nb.h_excluded_numbers, md_info.frc);
        // DEBUG DEBUG PRINTF
        float* temp_output;
        Malloc_Safely((void**)&(temp_output), sizeof(float) * 3 * 6);
        cudaMemcpy(temp_output, fgm_surf.d_E_result, sizeof(float) * 3 * 6, cudaMemcpyDeviceToHost);
        printf("Electric Field At 2rd H2O's Oxygen = %f,%f,%f\n", temp_output[9], temp_output[10], temp_output[11]);
        
        Main_Iteration();
        Main_Print();
    }
    fgm_surf.Potential_file_saver("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\CSR-Matrix\\result.txt"); // DEBUG
    fgm_surf.Sphere_saver("C:\\Users\\15653\\Desktop\\FGM-Python-Debug\\unit_492Sphere\\result.txt"); // DEBUG
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char* argv[])
{
    controller.Initial(argc, argv, SUBPACKAGE_HINT);
    cv_controller.Initial(&controller, &md_info.no_direct_interaction_virtual_atom_numbers);
    md_info.Initial(&controller);
    controller.Step_Print_Initial("potential", "%.2f");
    cv_controller.atom_numbers = md_info.atom_numbers;
    plugin.Initial(&md_info, &controller, &cv_controller, &neighbor_list);
    fgm_surf.Initial(&controller, md_info.atom_numbers);   // DEBUG-7-22 

    if (md_info.mode >= md_info.NVT && !controller.Command_Exist("thermostat"))
    {
        controller.Throw_SPONGE_Error(spongeErrorMissingCommand, "Main_Initial", "Reason:\n\tthermostat is required for NVT or NPT simulations\n");
    }
    if  (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "middle_langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "langevin"))
    {
        langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "andersen_thermostat"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature, md_info.atom_numbers, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.sys.target_temperature);
    }

    if (md_info.pbc.pbc)
    {
        lj.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
        lj_soft.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
        pairwise_force.Initial(&controller);
        solvent_lj.Initial(&controller, &lj, &lj_soft, md_info.res.residue_numbers, md_info.res.h_res_start, md_info.res.h_res_end, md_info.mode >= md_info.NVT);
        pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff);
        sits.Initial(&controller, md_info.atom_numbers);
        if (sits.is_initialized)
        {
            sits_dihedral.Initial(&controller, "sits_dihedral");
        }
        nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);
    }
    else
    {
        LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
        CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
        if (controller.Command_Exist("gb", "in_file"))
        {
            gb.Initial(&controller, md_info.nb.cutoff);
        }
        nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B, LJ_NOPBC.h_atom_LJ_type);
    }

    listed_forces.Initial(&controller, &md_info.sys.connectivity, &md_info.sys.connected_distance);

    bond.Initial(&controller, &md_info.sys.connectivity, &md_info.sys.connected_distance);
    bond_soft.Initial(&controller);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    cmap.Initial(&controller);

    hard_wall.Initial(&controller, md_info.sys.target_temperature, md_info.sys.target_pressure, md_info.mode == md_info.NPT);
    soft_walls.Initial(&controller, md_info.atom_numbers);

    restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);
    restrain_cv.Initial(&controller, &cv_controller);
    steer_cv.Initial(&controller, &cv_controller);
    meta.Initial(&controller, &cv_controller);

    if (controller.Command_Exist("constrain_mode"))
    {    
        constrain.Initial_List(&controller, md_info.sys.connectivity, md_info.sys.connected_distance, md_info.h_mass);
        if (middle_langevin.is_initialized)
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, middle_langevin.exp_gamma, 0, md_info.h_mass, &md_info.sys.freedom);
        else
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, 1.0, md_info.mode == md_info.MINIMIZATION, md_info.h_mass, &md_info.sys.freedom);
        if (!(controller.Command_Exist("settle_disable") && controller.Get_Bool("settle_disable", "Main_Initial")))
        {
            settle.Initial(&controller, &constrain, md_info.h_mass);
        }
        if (controller.Command_Choice("constrain_mode", "simple_constrain"))
        {
            simple_constrain.Initial_Simple_Constrain(&controller, &constrain);
        }
        if (controller.Command_Choice("constrain_mode", "shake"))
        {
            shake.Initial_Simple_Constrain(&controller, &constrain);
        }
    }

    if (md_info.mode == md_info.NPT && !controller.Command_Exist("barostat"))
    {
        controller.Throw_SPONGE_Error(spongeErrorMissingCommand, "Main_Initial", "Reason:\n\tbarostat is required for NPT simulations\n");
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers, md_info.sys.target_pressure, md_info.sys.box_length, md_info.res.is_initialized);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "berendsen_barostat"))
    {
        bd_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "andersen_barostat"))
    {
        ad_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }

    vatom.Initial(&controller, &cv_controller, md_info.atom_numbers, md_info.no_direct_interaction_virtual_atom_numbers,
        cv_controller.cv_vatom_name, md_info.h_mass, &md_info.sys.freedom, &md_info.sys.connectivity);
    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
    md_info.mol.Initial(&controller);
    md_info.mol.Molecule_Crd_Map();
    if (md_info.pbc.pbc)
    {
        neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
        neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
            md_info.nb.d_excluded_numbers, neighbor_list.FORCED_UPDATE);
    }
    cv_controller.Print_Initial();
    plugin.After_Initial();
    cv_controller.Input_Check();
    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
    controller.core_time.Start();
}

void Main_Clear()
{
    controller.core_time.Stop();
    controller.printf("Core Run Wall Time: %f seconds\n", controller.core_time.time);
    
    if (md_info.mode == md_info.MINIMIZATION)
    {
        controller.simulation_speed = md_info.sys.steps / controller.core_time.time;
        controller.printf("Core Run Speed: %f step/second\n", controller.simulation_speed);
    }
    else if (md_info.mode == md_info.RERUN)
    {
        controller.simulation_speed = md_info.sys.steps / controller.core_time.time;
        controller.printf("Core Run Speed: %f frame/second\n", controller.simulation_speed);
    }
    else
    {
        controller.simulation_speed = md_info.sys.steps * md_info.dt / CONSTANT_TIME_CONVERTION / controller.core_time.time * 86.4;
        controller.printf("Core Run Speed: %f ns/day\n", controller.simulation_speed);
    }
    fcloseall();

    if (controller.Command_Exist("end_pause"))
    {
        if (atoi(controller.Command("end_pause")) == 1)
        {
            printf("End Pause\n");
            getchar();
        }
    }
}

void Main_Calculate_Force()
{
    md_info.MD_Information_Crd_To_Uint_Crd();
    if (md_info.mode == md_info.RERUN)
    {
        return;
    }
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    if (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt)
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps, &md_info.need_potential);
    bd_baro.Ask_For_Calculate_Pressure(md_info.sys.steps, &md_info.need_pressure);
    ad_baro.Ask_For_Calculate_Pressure(md_info.sys.steps, &md_info.need_pressure);

    sits.Reset_Force_Energy(md_info.need_potential);
    
    if (sits.is_initialized)
    {
        sits_dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, sits.pw_select.select_force[0], sits.pw_select.select_atom_energy[0]);
      //  sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, &lj, md_info.frc,
      //      neighbor_list.d_nl, md_info.nb.cutoff, pme.beta, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
      //  sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, &lj_soft, md_info.frc,
      //      neighbor_list.d_nl, md_info.nb.cutoff, pme.beta, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    }
    else
    {
     lj.LJ_NOPME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
          neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
     lj_soft.LJ_Soft_Core_NOPME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers - solvent_lj.solvent_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
          neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    }
    solvent_lj.LJ_NOPME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers, md_info.res.residue_numbers, md_info.res.d_res_start, md_info.res.d_res_end, md_info.uint_crd, md_info.d_charge,
      md_info.frc, neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);

    lj.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial,
       md_info.need_potential, md_info.sys.d_potential);
    lj_soft.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial, md_info.need_potential, md_info.sys.d_potential);
    
    // pairwise_force.Compute_Force(neighbor_list.d_nl, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.nb.cutoff, pme.beta,
    //        md_info.d_charge, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);

    //pme.PME_Excluded_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.d_charge,
    //     md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, md_info.frc, pme.d_correction_atom_energy);

    //pme.PME_Reciprocal_Force_With_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.frc, 
    //    md_info.need_pressure, md_info.need_potential, md_info.sys.d_virial, md_info.sys.d_potential, md_info.sys.steps);


    LJ_NOPBC.LJ_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    CF_NOPBC.Coulomb_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.d_charge, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    gb.Get_Effective_Born_Radius(md_info.pbc.nopbc_crd);
    gb.GB_Force_With_Atom_Energy(md_info.atom_numbers, md_info.pbc.nopbc_crd, md_info.d_charge, md_info.frc, md_info.d_atom_energy);

    nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);

    bond.Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    bond_soft.Soft_Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    angle.Angle_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    improper.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    cmap.CMAP_Force_with_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    listed_forces.Compute_Force(md_info.crd, md_info.sys.box_length, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    soft_walls.Compute_Force(md_info.atom_numbers, md_info.crd, md_info.frc, md_info.d_atom_energy);

    plugin.Calculate_Force();

    restrain.Restraint(md_info.crd, md_info.sys.box_length, md_info.d_atom_energy, md_info.d_atom_virial, md_info.frc);
    restrain_cv.Restraint(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
                md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps, 
        md_info.sys.d_potential, md_info.sys.d_virial, md_info.frc, md_info.need_potential, md_info.need_pressure);
    steer_cv.Steer(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
                md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps, 
        md_info.sys.d_potential, md_info.sys.d_virial, md_info.frc, md_info.need_potential, md_info.need_pressure);
    meta.Do_Metadynamics(md_info.atom_numbers + md_info.no_direct_interaction_virtual_atom_numbers,
        md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length,
        md_info.sys.steps, md_info.need_potential, md_info.need_pressure, md_info.frc, md_info.sys.d_potential, md_info.sys.d_virial);

    sits.Update_And_Enhance(md_info.sys.steps, md_info.sys.d_potential, md_info.need_pressure, md_info.sys.d_virial, md_info.frc, 1.0 / (CONSTANT_kB * md_info.sys.target_temperature));

    vatom.Force_Redistribute(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc);
    md_info.Calculate_Pressure_And_Potential_If_Needed();
}

void Main_Iteration()
{
    if (md_info.mode == md_info.RERUN)
    {
        vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
        md_info.MD_Information_Crd_To_Uint_Crd();
        return;
    }

    if (mc_baro.is_initialized && md_info.sys.steps % mc_baro.update_interval == 0)
    {
        mc_baro.energy_old = md_info.sys.h_potential;
        cudaMemcpy(mc_baro.frc_backup, md_info.frc, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mc_baro.crd_backup, md_info.crd, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);

        mc_baro.Volume_Change_Attempt(md_info.sys.box_length);

        if (mc_baro.scale_coordinate_by_molecule)
        {
            md_info.mol.Molecule_Crd_Map(mc_baro.crd_scale_factor);
        }
        else
        {
            mc_baro.Scale_Coordinate_Atomically(md_info.atom_numbers, md_info.crd);
        }

        Main_Box_Length_Change(mc_baro.crd_scale_factor);

        Main_Calculate_Force();
        mc_baro.energy_new = md_info.sys.h_potential;

        if (mc_baro.scale_coordinate_by_molecule)
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.mol.molecule_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);
        else
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.atom_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);

        if (mc_baro.couple_dimension != mc_baro.NO && mc_baro.couple_dimension != mc_baro.XYZ)
            mc_baro.extra_term -= mc_baro.surface_number * mc_baro.surface_tension * mc_baro.DeltaS;

        mc_baro.accept_possibility = mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility = expf(-mc_baro.accept_possibility / (CONSTANT_kB * md_info.sys.target_temperature));

        if (mc_baro.Check_MC_Barostat_Accept())
        {
            mc_baro.crd_scale_factor = 1.0 / mc_baro.crd_scale_factor;
            cudaMemcpy(md_info.crd, mc_baro.crd_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
            Main_Box_Length_Change(mc_baro.crd_scale_factor);
            neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
            cudaMemcpy(md_info.frc, mc_baro.frc_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
            md_info.sys.h_potential = mc_baro.energy_old;
        }
        if ((!mc_baro.reject && (mc_baro.newV > 1.331 * mc_baro.V0 || mc_baro.newV < 0.729 * mc_baro.V0)))
        {
            Main_Volume_Change_Largely();
            mc_baro.V0 = mc_baro.newV;
        }
        mc_baro.Delta_Box_Length_Max_Update();
    }
    
    settle.Remember_Last_Coordinates(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    simple_constrain.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    shake.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);

    if (md_info.mode == md_info.NVE)
    {
        md_info.nve.Leap_Frog();
    }
    else if (md_info.mode == md_info.MINIMIZATION)
    {
        md_info.min.Gradient_Descent();
    }
    else if (middle_langevin.is_initialized)
    {
        middle_langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.vel, md_info.acc, md_info.crd);
    }
    else if (langevin.is_initialized)
    {
        langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.crd, md_info.vel, md_info.acc);
    }
    else if (bd_thermo.is_initialized)
    {
        bd_thermo.Record_Temperature(md_info.sys.Get_Atom_Temperature(), md_info.sys.freedom);
        md_info.nve.Leap_Frog();
        bd_thermo.Scale_Velocity(md_info.atom_numbers, md_info.vel);
    }
    else if (ad_thermo.is_initialized)
    {
        if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
        {
            ad_thermo.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt);
            constrain.v_factor = FLT_MIN;
            constrain.x_factor = 0.5;
        }
        else
        {
            md_info.nve.Leap_Frog();
            constrain.v_factor = 1.0;
            constrain.x_factor = 1.0;
        }
    }
    else if (nhc.is_initialized)
    {
        nhc.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt, md_info.sys.Get_Total_Atom_Ek(), md_info.sys.freedom);
    }

    settle.Do_SETTLE(md_info.d_mass, md_info.crd, md_info.sys.box_length, md_info.vel, md_info.need_pressure, md_info.sys.d_pressure);
    simple_constrain.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    shake.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    

    if (bd_baro.is_initialized && md_info.sys.steps % bd_baro.update_interval == 0)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        bd_baro.crd_scale_factor = 1 - bd_baro.update_interval * bd_baro.compressibility * bd_baro.dt / bd_baro.taup / 3 * (md_info.sys.target_pressure - md_info.sys.h_pressure);
        if (bd_baro.stochastic_term)
        {
            bd_baro.crd_scale_factor += sqrtf(2 * CONSTANT_kB * md_info.sys.target_temperature* bd_baro.compressibility / bd_baro.taup / bd_baro.newV)
                / 3 * bd_baro.n(bd_baro.e);
            Scale_List((float*)md_info.vel, 1.0f / bd_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        }
        md_info.Scale_Position_To_Center(bd_baro.crd_scale_factor);
        Main_Volume_Change(bd_baro.crd_scale_factor);
        bd_baro.newV = md_info.sys.Get_Volume();
        if (bd_baro.newV > 1.331 * bd_baro.V0 || bd_baro.newV < 0.729 * bd_baro.V0)
        {
            Main_Volume_Change_Largely();
            bd_baro.V0 = bd_baro.newV;
        }
    }

    if (ad_baro.is_initialized && md_info.sys.steps % ad_baro.update_interval == 0)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        ad_baro.dV_dt += (md_info.sys.h_pressure - md_info.sys.target_pressure) * ad_baro.h_mass_inverse * md_info.dt * ad_baro.update_interval;
        ad_baro.Control_Velocity_Of_Box(md_info.dt * ad_baro.update_interval, md_info.sys.target_temperature);
        ad_baro.new_V = md_info.sys.Get_Volume() + ad_baro.dV_dt * md_info.dt * ad_baro.update_interval;
        ad_baro.crd_scale_factor = cbrt(ad_baro.new_V / md_info.sys.Get_Volume());
        md_info.Scale_Position_To_Center(ad_baro.crd_scale_factor);
        Scale_List((float*)md_info.vel, 1.0f / ad_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        Main_Volume_Change(ad_baro.crd_scale_factor);

        if (ad_baro.new_V > 1.331 * ad_baro.V0 || ad_baro.new_V < 0.729 * ad_baro.V0)
        {
            Main_Volume_Change_Largely();
            ad_baro.V0 = ad_baro.new_V;
        }
    }

    if (md_info.mode == md_info.MINIMIZATION)
    {
        md_info.min.Check_Nan();
    }

    hard_wall.Reflect(md_info.atom_numbers, md_info.crd, md_info.vel);
    md_info.mol.Molecule_Crd_Map();
    md_info.MD_Information_Crd_To_Uint_Crd();

    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);
    md_info.MD_Information_Crd_To_Uint_Crd();
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
}

void Main_Print()
{
    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0)
    {
        md_info.Step_Print(&controller);
        if (!md_info.pbc.pbc)
        {
            controller.Step_Print("Coulomb", CF_NOPBC.Get_Energy(md_info.pbc.nopbc_crd, md_info.d_charge, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers), true);
            controller.Step_Print("gb", gb.Get_Energy(md_info.pbc.nopbc_crd, md_info.d_charge), true);
            controller.Step_Print("LJ", LJ_NOPBC.Get_Energy(md_info.pbc.nopbc_crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers), true);
        }
        else if (!sits.is_initialized)
        {
            controller.Step_Print("LJ", lj.Get_Energy(md_info.uint_crd, neighbor_list.d_nl, pme.beta, md_info.d_charge, pme.d_direct_atom_energy), true);
            controller.Step_Print("LJ_soft", lj_soft.Get_Energy(md_info.uint_crd, neighbor_list.d_nl, pme.beta, md_info.d_charge, pme.d_direct_atom_energy), true);
        }
        else
        {
            sits_dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, false);
            sits.Step_Print(&controller, 1.0 / (CONSTANT_kB * md_info.sys.target_temperature),
                &lj, &lj_soft, md_info.atom_numbers, md_info.uint_crd, neighbor_list.d_nl,
                pme.beta, md_info.d_charge, pme.d_direct_atom_energy, md_info.sys.steps, sits_dihedral.d_sigma_of_dihedral_ene);
        }
        controller.Step_Print(pairwise_force.force_name.c_str(),
            pairwise_force.Get_Energy(neighbor_list.d_nl, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.nb.cutoff,
                pme.beta, md_info.d_charge, pme.d_direct_atom_energy),true);
        soft_walls.Step_Print(&controller, md_info.atom_numbers, md_info.crd);
        pme.Step_Print(&controller, md_info.uint_crd, md_info.d_charge, neighbor_list.d_nl, md_info.pbc.uint_dr_to_dr_cof,
            md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
        controller.Step_Print("nb14_LJ", nb14.Get_14_LJ_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("nb14_EE", nb14.Get_14_CF_Energy(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("bond", bond.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("bond_soft", bond_soft.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        listed_forces.Step_Print(&controller, md_info.crd, md_info.sys.box_length);
        controller.Step_Print("angle", angle.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("urey_bradley", urey_bradley.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("restrain", restrain.Get_Energy(md_info.crd, md_info.sys.box_length), true);
        controller.Step_Print("dihedral", dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("improper_dihedral", improper.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("cmap", cmap.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof), true);
        controller.Step_Print("density", md_info.sys.Get_Density());
        controller.Step_Print("pressure", md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION);
        controller.Step_Print("dV_dt", ad_baro.dV_dt);
        controller.Step_Print("sits_dihedral", sits_dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        if (meta.is_initialized)
        {
            meta.cv->Compute(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, CV_NEED_CPU_VALUE, md_info.sys.steps + 1);
            controller.Step_Print("meta1d", meta.Potential(meta.cv->value), true);
        }
        controller.Step_Print(restrain_cv.module_name, 
            restrain_cv.Get_Energy(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps + 1),
            true);
        controller.Step_Print(steer_cv.module_name, 
            steer_cv.Get_Energy(md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length, md_info.sys.steps + 1),
            true);
        cv_controller.Step_Print(md_info.sys.steps, md_info.atom_numbers, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd, md_info.sys.box_length);
        plugin.Mdout_Print();

        controller.Step_Print("potential", controller.printf_sum);
        controller.Print_To_Screen_And_Mdout();
        neighbor_list.Check_Overflow(&controller);
        controller.Check_Error(md_info.sys.h_potential);
        if (md_info.mode == md_info.RERUN)
        {
            md_info.rerun.Iteration();
            Main_Box_Length_Change(md_info.rerun.box_length_change_factor);
        }
    }
    if (md_info.output.write_trajectory_interval && md_info.sys.steps % md_info.output.write_trajectory_interval == 0)
    {
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Box_Traj_File();
        meta.Write_Potential();
        if (md_info.output.is_vel_traj)
        {
            md_info.output.Append_Vel_Traj_File();
        }
        if (md_info.output.is_frc_traj)
        {
            md_info.output.Append_Frc_Traj_File();
        }
        nhc.Save_Trajectory_File();
    }
    if (md_info.output.write_restart_file_interval && md_info.sys.steps % md_info.output.write_restart_file_interval == 0)
    {
        md_info.output.Export_Restart_File();
        nhc.Save_Restart_File();
    }
}

void Main_Volume_Change(double factor)
{
    md_info.Update_Volume(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Volume(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    md_info.mol.Molecule_Crd_Map();
}

void Main_Box_Length_Change(VECTOR factor)
{
    md_info.Update_Box_Length(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list,
        md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Box_Length(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    md_info.mol.Molecule_Crd_Map();
}

void Main_Volume_Change_Largely()
{
    controller.printf("Some modules are based on the meshing methods, and it is more precise to re-initialize these modules now for a long time or a large volume change.\n");
    neighbor_list.Clear();
    pme.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, 1);
    pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length ,md_info.nb.cutoff );
    controller.printf("------------------------------------------------------------------------------------------------------------\n"); 
}



