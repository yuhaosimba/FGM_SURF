/*
* Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
*
* NOTICE TO LICENSEE:
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#ifndef SOLVENT_LENNARD_JONES_FORCE_CUH
#define SOLVENT_LENNARD_JONES_FORCE_CUH
#include "../common.cuh"
#include "../control.cuh"
#include "Lennard_Jones_force.cuh"
#include "LJ_soft_core.cuh"

struct SOLVENT_LENNARD_JONES
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20230516;

    int solvent_numbers = 0;
    int solvent_start = 0;
    int water_points = 3;

    LENNARD_JONES_INFORMATION* lj_info;
    LJ_SOFT_CORE* lj_soft_info;
    UINT_VECTOR_LJ_TYPE* soft_to_hard_crd;
    void Initial(CONTROLLER* controller, LENNARD_JONES_INFORMATION* lj, LJ_SOFT_CORE* lj_soft, int res_num, int* res_start, int* res_end, bool default_enable, const char* module_name = NULL);
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const int residue_numbers, const int* d_res_start, const int* d_res_end, 
        const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
        const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
        const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy);
    void LJ_NOPME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const int residue_numbers, const int* d_res_start, const int* d_res_end, const UNSIGNED_INT_VECTOR* uint_crd,
        const float* charge, VECTOR* frc, const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
        const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy);
};

#endif