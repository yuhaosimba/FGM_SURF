/*
* Copyright 2021-2023 Gao's lab, Peking University, CCME. All rights reserved.
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

#ifndef PAIRWISE_FORCE_CUH
#define PAIRWISE_FORCE_CUH
#include "../common.cuh"
#include "../control.cuh"

struct PAIRWISE_FORCE
{
    char module_name[CHAR_LENGTH_MAX];
    bool is_initialized = false;
    bool is_controller_printf_initialized = false;
    int last_modify_date = 20231031;

    int atom_numbers = 0;
    int type_numbers = 0;
    bool with_ele = true;

    std::string force_name;
    int n_ij_parameter = 0;
    std::vector<std::string> parameter_type;
    std::vector<std::string> parameter_name;
    std::string source_code;
    std::string ele_code;
    JIT_Function force_function;

    void** gpu_parameters;
    void** cpu_parameters;
    int* cpu_pairwise_types;
    int* gpu_pairwise_types;
    std::vector<void*> launch_args;

    float* item_energy;
    float* sum_energy;

    void Initial(CONTROLLER* controller, const char* module_name = NULL);
    void Read_Configuration(CONTROLLER* controller);
    void JIT_Compile(CONTROLLER* controller);
    void Real_Initial(CONTROLLER* controller);
    void Compute_Force(ATOM_GROUP* nl, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, float cutoff,
        float pme_beta, float* charge, VECTOR* frc, int need_energy, float* atom_energy,
        int need_virial, float* atom_virial, float* pme_direct_atom_energy);
    float Get_Energy(ATOM_GROUP* nl, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, float cutoff,
        float pme_beta, float* charge, float* pme_direct_atom_energy);
};



#endif //BOND (bond.cuh)
