#include "pairwise_force.cuh"

void PAIRWISE_FORCE::Initial(CONTROLLER* controller, const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "pairwise_force");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (controller->Command_Exist(this->module_name, "in_file"))
    {
        controller->printf("START INITIALIZING PAIRWISE FORCE:\n");
        this->Read_Configuration(controller);
        this->JIT_Compile(controller);
        this->Real_Initial(controller);
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->force_name.c_str(), "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    if (is_initialized)
    {
        controller[0].printf("END INITIALIZING PAIRWISE FORCE\n\n");
    }
    else
    {
        controller->printf("PAIRWISE FORCE IS NOT INITIALIZED\n\n");
    }
}

void PAIRWISE_FORCE::Read_Configuration(CONTROLLER* controller)
{
    Configuration_Reader cfg;
    cfg.Open(controller->Command(this->module_name, "in_file"));
    cfg.Close();
    if (!cfg.error_reason.empty())
    {
        cfg.error_reason = "Reason:\n\t" + cfg.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", cfg.error_reason.c_str());
    }
    if (cfg.sections.size() > 1)
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", "Reason:\n\tOnly one pairwise force can be used\n");
    }
    force_name = cfg.sections[0];
    if (!cfg.Key_Exist(force_name, "potential"))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", string_format("Reason:\n\tThe potential of the pairwise force %FORCE% is required ([[ potential ]])\n", { {"FORCE", force_name} }).c_str());
    }
    source_code = cfg.Get_Value(force_name, "potential");
    if (!cfg.Key_Exist(force_name, "parameters"))
    {
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", string_format("Reason:\n\tThe parameters of the pairwise force %FORCE% are required ([[ parameter ]])\n", { {"FORCE", force_name} }).c_str());
    }
    std::string parameter_strings = cfg.Get_Value(force_name, "parameters");
    std::vector<std::string> parameter_and_types = string_split(parameter_strings, ",");
    for (std::string s : parameter_and_types)
    {
        std::vector<std::string> parameter_and_type = string_split(string_strip(s), " ");
        if (parameter_and_type[0] != "int" && parameter_and_type[0] != "float")
        {
            controller->Throw_SPONGE_Error(spongeErrorTypeErrorCommand, "PAIRWISE_FORCE::Initialize_Parameters", "Reason:\n\tOnly 'int' or 'float' parameter is acceptable\n");
        }
        this->parameter_type.push_back(parameter_and_type[0]);
        this->parameter_name.push_back(parameter_and_type[1]);
    }
    n_ij_parameter = 0;
    for (auto s : this->parameter_name)
    {
        if (s.rfind("_ij") == s.length() - 3)
        {
            n_ij_parameter -= 1;
        }
        else if (n_ij_parameter < 0)
        {
            n_ij_parameter *= -1;
        }
        else
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise parameters should be placed in front of atomic parameters");
        }
    }
    n_ij_parameter = abs(n_ij_parameter);
    with_ele = true;
    if (cfg.Key_Exist(force_name, "with_ele"))
    {
        ele_code = cfg.Get_Value(force_name, "with_ele");
        if (!is_str_equal(ele_code.c_str(), "true") && !is_str_equal(ele_code.c_str(), "false") && !is_str_int(ele_code.c_str()))
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand, "PAIRWISE_FORCE::Initialize_Parameters",
                "Reason:\n\tPairwise [[ with_ele ]] should be 'true', 'false' or integers (0 for 'false' and others for 'true')");
        }
        if (is_str_equal(ele_code.c_str(), "true") || (is_str_int(ele_code.c_str()) && atoi(ele_code.c_str())))
        {
            with_ele = true;
        }
        else
        {
            with_ele = false;
        }
    }
    if (with_ele)
    {
        ele_code = "E_ele = charge_i * charge_j * erfc(beta * r_ij) / r_ij;";
        if (cfg.Key_Exist(force_name, "electrostatic_potential"))
        {
            ele_code = cfg.Get_Value(force_name, "electrostatic_potential");
        }
    }
    for (auto s : cfg.value_unused)
    {
        std::string error_reason = string_format("Reason:\n\t[[ %s% ]] should not be one of the keys of the pairwise force input file",
            { {"s", s.second} });
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Read_Configuration", error_reason.c_str());
    }
}

void PAIRWISE_FORCE::JIT_Compile(CONTROLLER* controller)
{
    if (with_ele)
        controller->printf("        %s will be calculated with direct part of the electrostatic potential\n", this->force_name.c_str());
    else
        controller->printf("        %s will not be calculated with direct part of the electrostatic potential\n", this->force_name.c_str());
    std::string full_source_code = R"JIT(#include "common.cuh"
__device__ __forceinline__ int Get_Pairwise_Type(int a, int b)
{
    int y = (b - a);
    int x = y >> 31;
    y = (y ^ x) - x;
    x = b + a;
    int z = (x + y) >> 1;
    x = (x - y) >> 1;
    return (z * (z + 1) >> 1) + x;
}

extern "C" __global__ void pairwise_force_energy_and_virial(%PARM_ARGS%,
    const float* charge, const float pme_beta, ATOM_GROUP* nl, const int* pairwise_types,
    const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, const float cutoff,
    VECTOR* frc, float* atom_energy, float* atom_virial, float* pme_atom_energy, int atom_numbers)
{
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < atom_numbers)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i];
        int pairwise_type_i = pairwise_types[atom_i];
        VECTOR frc_record = { 0.0f, 0.0f, 0.0f };
        float virial_lj = 0.0f;
        float energy_total = 0.0f;
        float energy_coulomb = 0.0f;
        float charge_i = 0;
        float charge_j = 0;
        if (pme_atom_energy != NULL)
        {
            charge_i = charge[atom_i];
        }
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            VECTOR vector_dr = Get_Periodic_Displacement(uint_crd[atom_j], r1, scaler);
            float float_dr_ij = norm3df(vector_dr.x, vector_dr.y, vector_dr.z);
            if (float_dr_ij < cutoff)
            {
                int atom_pairwise_type = Get_Pairwise_Type(pairwise_type_i, pairwise_types[atom_j]);
                %PARM_DEC%
                SADfloat<1> r_ij(float_dr_ij, 0);
                SADfloat<1> E, E_ele;
                %SOURCE_CODE%
                energy_total += E.val;
                if (pme_atom_energy != NULL)
                {
                    charge_j = charge[atom_j];
                    %COULOMB_CODE%
                    energy_coulomb += E_ele.val;
                }
                if (frc != NULL)
                {
                    float frc_abs = E.dval[0] / float_dr_ij;
                    if (pme_atom_energy != NULL)
                    {
                        frc_abs += E_ele.dval[0] / float_dr_ij;
                    }
                    VECTOR frc_temp = frc_abs * vector_dr;
                    frc_record = frc_record + frc_temp;
                    atomicAdd(frc + atom_j, -frc_temp);
                }
                if (atom_virial != NULL)
                {
                    virial_lj -=  E.dval[0] * float_dr_ij;
                }
            }
        }
        if (frc != NULL)
        {
            Warp_Sum_To(frc + atom_i, frc_record);
        }
        if (pme_atom_energy != NULL && (atom_energy != NULL || atom_virial != NULL))
        {
            Warp_Sum_To(pme_atom_energy + atom_i, energy_coulomb);
        }
        if (atom_energy != NULL)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total);
        }
        if (atom_virial != NULL)
        {
            Warp_Sum_To(atom_virial + atom_i, virial_lj);
        }
    }
}
)JIT";
    std::string PARM_ARGS = string_join("const %0%* %1%_list", ", ", { parameter_type, parameter_name });
    std::string PARM_DEC = string_join("                const %0% %1% = %1%_list[atom_pairwise_type];", "\n", { parameter_type, parameter_name });
    full_source_code = string_format(full_source_code, { {"PARM_ARGS", PARM_ARGS}, {"PARM_DEC", PARM_DEC}, {"SOURCE_CODE", source_code}, {"COULOMB_CODE", ele_code} });
    force_function.Compile(full_source_code);
    if (!force_function.error_reason.empty())
    {
        force_function.error_reason = "Reason:\n" + force_function.error_reason;
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed, "PAIRWISE_FORCE::JIT_Compile", force_function.error_reason.c_str());
    }
}

void PAIRWISE_FORCE::Real_Initial(CONTROLLER* controller)
{
    FILE* fp;
    if (!controller->Command_Exist(this->force_name.c_str(), "in_file"))
    {
        std::string error_reason = "Reason:\n\tlisted force '" + this->force_name + "' is defined, but " + this->force_name + "_in_file is not provided\n";
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "PAIRWISE_FORCE::Initial", error_reason.c_str());
    }
    controller->printf("    Initializing %s\n", this->force_name.c_str());
    Open_File_Safely(&fp, controller->Command(this->force_name.c_str(), "in_file"), "r");
    if (fscanf(fp, "%d %d", &atom_numbers, &type_numbers) != 2)
    {
        std::string error_reason = "Reason:\n\tFail to read the number of atoms and/or types of the pairwise force '" + this->force_name + "'\n";
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", error_reason.c_str());
    }
    int total_type_pairwise_numbers = type_numbers * (type_numbers + 1) / 2;
    Malloc_Safely((void**)&cpu_parameters, sizeof(void*) * parameter_name.size());
    Malloc_Safely((void**)&gpu_parameters, sizeof(void*) * parameter_name.size());
    launch_args = std::vector<void*>(parameter_name.size() + 12);
    Malloc_Safely((void**)&cpu_pairwise_types, sizeof(int) * atom_numbers);
    Cuda_Malloc_Safely((void**)&item_energy, sizeof(float) * atom_numbers);
    Cuda_Malloc_Safely((void**)&sum_energy, sizeof(float));
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Malloc_Safely((void**)cpu_parameters + j, sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Malloc_Safely((void**)cpu_parameters + j, sizeof(float) * total_type_pairwise_numbers);
        }
        launch_args[j] = gpu_parameters + j;
    }
    for (int j = 0; j < n_ij_parameter; j++)
    {
        for (int i = 0; i < total_type_pairwise_numbers; i++)
        {
            int scanf_ret = 0;
            if (parameter_type[j] == "int")
            {
                scanf_ret = fscanf(fp, "%d", ((int*)cpu_parameters[j]) + i);
            }
            else
            {
                scanf_ret = fscanf(fp, "%f", ((float*)cpu_parameters[j]) + i);
            }
            if (scanf_ret == 0)
            {
                std::string error_reason = "Reason:\n\tFail to read the parameters of the pairwise force '" + this->force_name + "'\n";
                controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", error_reason.c_str());
            }
        }
    }
    for (int i = 0; i < atom_numbers; i++)
    {
        if (fscanf(fp, "%d", cpu_pairwise_types + i) != 1)
        {
            std::string error_reason = "Reason:\n\tFail to read the types of the pairwise force '" + this->force_name + "'\n";
            controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, "PAIRWISE_FORCE::Initial", error_reason.c_str());
        }
    }
    fclose(fp);
    for (int j = 0; j < n_ij_parameter; j++)
    {
        if (parameter_type[j] == "int")
        {
            Cuda_Malloc_And_Copy_Safely((void**)gpu_parameters + j, cpu_parameters[j], sizeof(int) * total_type_pairwise_numbers);
        }
        else
        {
            Cuda_Malloc_And_Copy_Safely((void**)gpu_parameters + j, cpu_parameters[j], sizeof(float) * total_type_pairwise_numbers);
        }
    }
    Cuda_Malloc_And_Copy_Safely((void**)&gpu_pairwise_types, cpu_pairwise_types, sizeof(int) * atom_numbers);
    this->is_initialized = 1;
}

void PAIRWISE_FORCE::Compute_Force(ATOM_GROUP* nl, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, float cutoff,
    float pme_beta, float* charge, VECTOR* frc,
    int need_energy, float* atom_energy, int need_virial, float* atom_virial, float* pme_direct_atom_energy)
{
    if (!this->is_initialized)
        return;

    cudaMemset(this->item_energy, 0, sizeof(float) * atom_numbers);
    float* NULLPTR = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types;
    launch_args[parameter_name.size() + 4] = &uint_crd;
    launch_args[parameter_name.size() + 5] = &scaler;
    launch_args[parameter_name.size() + 6] = &cutoff;
    launch_args[parameter_name.size() + 7] = &frc;
    if (need_energy)
    {
        launch_args[parameter_name.size() + 8] = &atom_energy;
    }
    else
    {
        launch_args[parameter_name.size() + 8] = &NULLPTR;
    }
    if (need_virial)
    {
        launch_args[parameter_name.size() + 9] = &atom_virial;
    }
    else
    {
        launch_args[parameter_name.size() + 9] = &NULLPTR;
    }
    if (this->with_ele)
    {
        launch_args[parameter_name.size() + 10] = &pme_direct_atom_energy;
        cudaMemset(pme_direct_atom_energy, 0, sizeof(float) * this->atom_numbers);
    }
    else
    {
        launch_args[parameter_name.size() + 10] = &NULLPTR;
    }
    launch_args[parameter_name.size() + 11] = &atom_numbers;

    force_function({ (atom_numbers + 31u) / 32u, 1, 1 }, { 32, 32, 1 }, 0, 0, launch_args);
}

float PAIRWISE_FORCE::Get_Energy(ATOM_GROUP* nl, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler, float cutoff,
    float pme_beta, float* charge, float* pme_direct_atom_energy)
{
    if (!this->is_initialized)
        return 0;

    cudaMemset(this->item_energy, 0, sizeof(float) * atom_numbers);
    float* NULLPTR = NULL;
    launch_args[parameter_name.size()] = &charge;
    launch_args[parameter_name.size() + 1] = &pme_beta;
    launch_args[parameter_name.size() + 2] = &nl;
    launch_args[parameter_name.size() + 3] = &gpu_pairwise_types;
    launch_args[parameter_name.size() + 4] = &uint_crd;
    launch_args[parameter_name.size() + 5] = &scaler;
    launch_args[parameter_name.size() + 6] = &cutoff;
    launch_args[parameter_name.size() + 7] = &NULLPTR;
    launch_args[parameter_name.size() + 8] = &item_energy;
    launch_args[parameter_name.size() + 9] = &NULLPTR;
    if (this->with_ele)
    {
        launch_args[parameter_name.size() + 10] = &pme_direct_atom_energy;
        cudaMemset(pme_direct_atom_energy, 0, sizeof(float) * this->atom_numbers);
    }
    else
    {
        launch_args[parameter_name.size() + 10] = &NULLPTR;
    }
    launch_args[parameter_name.size() + 11] = &atom_numbers;
    CUresult res = force_function({ (atom_numbers + 31u) / 32u, 1, 1 }, { 32, 32, 1 }, 0, 0, launch_args);
    Sum_Of_List(item_energy, sum_energy, atom_numbers);
    float h_energy = NAN;
    if (res == CUDA_SUCCESS)
    {
        cudaMemcpy(&h_energy, sum_energy, sizeof(float), cudaMemcpyDeviceToHost);
    }
    return h_energy;
}