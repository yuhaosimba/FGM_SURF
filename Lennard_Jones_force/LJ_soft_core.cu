#include "LJ_soft_core.cuh"


__global__ void Copy_LJ_Type_And_Mask_To_New_Crd(const int atom_numbers, UINT_VECTOR_LJ_FEP_TYPE *new_crd, const int *LJ_type_A, const int * LJ_type_B, const int * mask)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].LJ_type = LJ_type_A[atom_i];
        new_crd[atom_i].LJ_type_B = LJ_type_B[atom_i];
        new_crd[atom_i].mask = mask[atom_i];
    }
}

static __global__ void device_add(float *variable, const float adder)
{
    variable[0] += adder;
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_FEP_TYPE *new_crd, const float *charge)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].uint_x = crd[atom_i].uint_x;
        new_crd[atom_i].uint_y = crd[atom_i].uint_y;
        new_crd[atom_i].uint_z = crd[atom_i].uint_z;
        new_crd[atom_i].charge = charge[atom_i];
    }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR* crd, UINT_VECTOR_LJ_FEP_TYPE* new_crd, const float* charge, const float* charge_BA)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].uint_x = crd[atom_i].uint_x;
        new_crd[atom_i].uint_y = crd[atom_i].uint_y;
        new_crd[atom_i].uint_z = crd[atom_i].uint_z;
        new_crd[atom_i].charge = charge[atom_i];
        new_crd[atom_i].charge_BA = charge_BA[atom_i];
    }
}
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_FEP_TYPE *new_crd)
{
    int atom_i = blockDim.x*blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        new_crd[atom_i].uint_x = crd[atom_i].uint_x;
        new_crd[atom_i].uint_y = crd[atom_i].uint_y;
        new_crd[atom_i].uint_z = crd[atom_i].uint_z;
    }
}

static __global__ void Total_C6_Get(int atom_numbers, int * atom_lj_type_A, int * atom_lj_type_B, float * d_lj_Ab, float * d_lj_Bb,float * d_factor, const float lambda)
{
    int i, j;
    float temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    float lambda_ = 1.0 - lambda;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
    {
        itype_A = atom_lj_type_A[i];
        itype_B = atom_lj_type_B[i];
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers; j += gridDim.y * blockDim.y)
        {
            jtype_A = atom_lj_type_A[j];
            jtype_B = atom_lj_type_B[j];
            yA = (jtype_A - itype_A);
            xA = yA >> 31;
            yA = (yA^xA) - xA;
            xA = jtype_A + itype_A;
            jtype_A = (xA + yA) >> 1;
            xA = (xA - yA) >> 1;
            atom_pair_LJ_type_A = (jtype_A*(jtype_A + 1) >> 1) + xA;

            yB = (jtype_B - itype_B);
            xB = yB >> 31;
            yB = (yB^xB) - xB;
            xB = jtype_B + itype_B;
            jtype_B = (xB + yB) >> 1;
            xB = (xB - yB) >> 1;
            atom_pair_LJ_type_B = (jtype_B*(jtype_B + 1) >> 1) + xB;
            
            temp_sum += lambda_ * d_lj_Ab[atom_pair_LJ_type_A];
            temp_sum += lambda * d_lj_Bb[atom_pair_LJ_type_B];
        }
    }
    atomicAdd(d_factor, temp_sum);
}

static __global__ void Total_C6_B_A_Get(int atom_numbers, int * atom_lj_type_A, int * atom_lj_type_B, float * d_lj_Ab, float * d_lj_Bb,float * d_factor)
{
    int i, j;
    float temp_sum = 0.0;
    int xA, yA, xB, yB;
    int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x)
    {
        itype_A = atom_lj_type_A[i];
        itype_B = atom_lj_type_B[i];
        for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers; j += gridDim.y * blockDim.y)
        {
            jtype_A = atom_lj_type_A[j];
            jtype_B = atom_lj_type_B[j];
            yA = (jtype_A - itype_A);
            xA = yA >> 31;
            yA = (yA^xA) - xA;
            xA = jtype_A + itype_A;
            jtype_A = (xA + yA) >> 1;
            xA = (xA - yA) >> 1;
            atom_pair_LJ_type_A = (jtype_A*(jtype_A + 1) >> 1) + xA;

            yB = (jtype_B - itype_B);
            xB = yB >> 31;
            yB = (yB^xB) - xB;
            xB = jtype_B + itype_B;
            jtype_B = (xB + yB) >> 1;
            xB = (xB - yB) >> 1;
            atom_pair_LJ_type_B = (jtype_B*(jtype_B + 1) >> 1) + xB;
            
            temp_sum += d_lj_Bb[atom_pair_LJ_type_B] - d_lj_Ab[atom_pair_LJ_type_A];
        }
    }
    atomicAdd(d_factor, temp_sum);
}

template<bool need_force, bool need_energy, bool need_virial, bool need_coulomb, bool need_du_dlambda>
static __global__ void Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA(const int atom_numbers, const ATOM_GROUP* nl,
    const UINT_VECTOR_LJ_FEP_TYPE* uint_crd, const VECTOR boxlength,
    const float* LJ_type_AA, const float* LJ_type_AB, const float* LJ_type_BA, const float* LJ_type_BB, const float cutoff,
    VECTOR* frc, const float pme_beta, float* atom_energy, float* atom_lj_virial, float* atom_direct_cf_energy, float* atom_du_dlambda_lj, float * atom_du_dlambda_direct,
    const float lambda, const float alpha, const float p, const float input_sigma_6, const float input_sigma_6_min)
{
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    float lambda_ = 1.0 - lambda;
    float alpha_lambda_p = alpha * powf(lambda, p);
    float alpha_lambda__p = alpha * powf(lambda_, p);
    if (atom_i < atom_numbers)
    {
        ATOM_GROUP nl_i = nl[atom_i];
        UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i];
        VECTOR frc_record = { 0., 0., 0. };
        float virial_lj = 0.;
        float energy_total = 0.;
        float energy_coulomb = 0.;
        float du_dlambda_lj = 0.;
        float du_dlambda_direct = 0.;
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_FEP_TYPE r2 = uint_crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                int atom_pair_LJ_type_B = Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                float AA = LJ_type_AA[atom_pair_LJ_type_A];
                float AB = LJ_type_AB[atom_pair_LJ_type_A];                
                float BA = LJ_type_BA[atom_pair_LJ_type_B];
                float BB = LJ_type_BB[atom_pair_LJ_type_B];
                if (BA * AA != 0 || BA + AA == 0)
                {
                    if (need_force)
                    {
                        float frc_abs = lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_coulomb && (need_energy || need_virial))
                    {
                        energy_coulomb += Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_energy)
                    {
                        energy_total += lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB);
                    }
                    if (need_virial)
                    {
                        virial_lj += lambda_ * Get_LJ_Virial(r1, r2, dr_abs, AA, AB) + lambda * Get_LJ_Virial(r1, r2, dr_abs, BA, BB);
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_abs, BA, BB) - Get_LJ_Energy(r1, r2, dr_abs, AA, AB);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_abs, pme_beta);
                        }
                    }
                }
                else
                {
                    float sigma_A = Get_Soft_Core_Sigma(AA, AB, input_sigma_6, input_sigma_6_min);
                    float sigma_B = Get_Soft_Core_Sigma(BA, BB, input_sigma_6, input_sigma_6_min);
                    float dr_softcore_A = Get_Soft_Core_Distance(AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                    float dr_softcore_B = Get_Soft_Core_Distance(BB, BA, sigma_B, dr_abs, alpha, p, 1 - lambda);
                    if (need_force)
                    {
                        float frc_abs = lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs, dr_softcore_A, AA, AB) 
                            + lambda * Get_Soft_Core_LJ_Force(r1, r2, dr_abs, dr_softcore_B, BA, BB);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = lambda_ * Get_Soft_Core_Direct_Coulomb_Force(r1, r2, dr_abs, dr_softcore_A, pme_beta)
                                + lambda * Get_Soft_Core_Direct_Coulomb_Force(r1, r2, dr_abs, dr_softcore_B, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record = frc_record + frc_lin;
                        atomicAdd(frc + atom_j, -frc_lin);
                    }
                    if (need_coulomb && (need_energy || need_virial))
                    {
                        energy_coulomb += lambda_ * Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A, pme_beta)
                            + lambda * Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B, pme_beta);
                    }
                    if (need_energy)
                    {
                        energy_total += lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB)
                            + lambda * Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB);
                    }
                    if (need_virial)
                    {
                        virial_lj += lambda_ * Get_Soft_Core_LJ_Virial(r1, r2, dr_abs, dr_softcore_A, AA, AB)
                            + lambda * Get_Soft_Core_LJ_Virial(r1, r2, dr_abs, dr_softcore_B, BA, BB);
                    }
                    if (need_du_dlambda)
                    {
                        du_dlambda_lj += Get_LJ_Energy(r1, r2, dr_softcore_B, BA, BB)
                            - Get_LJ_Energy(r1, r2, dr_softcore_A, AA, AB);
                        du_dlambda_lj += Get_Soft_Core_dU_dlambda(Get_LJ_Force(r1, r2, dr_softcore_A, AA, AB), sigma_A, dr_softcore_A, alpha, p, lambda)
                            - Get_Soft_Core_dU_dlambda(Get_LJ_Force(r1, r2, dr_softcore_B, BA, BB), sigma_B, dr_softcore_B, alpha, p, lambda_);
                        if (need_coulomb)
                        {
                            du_dlambda_direct += Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_B, pme_beta)
                                - Get_Direct_Coulomb_Energy(r1, r2, dr_softcore_A, pme_beta);
                            du_dlambda_direct += Get_Soft_Core_dU_dlambda(Get_Direct_Coulomb_Force(r1, r2, dr_softcore_B, pme_beta), sigma_B, dr_softcore_B, alpha, p, lambda_)
                                - Get_Soft_Core_dU_dlambda(Get_Direct_Coulomb_Force(r1, r2, dr_softcore_A, pme_beta), sigma_A, dr_softcore_A, alpha, p, lambda);
                            du_dlambda_direct += lambda * Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_softcore_B, pme_beta) 
                                + lambda_ * Get_Direct_Coulomb_dU_dlambda(r1, r2, dr_softcore_A, pme_beta); 
                        }
                    }
                }
            }
        }
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, frc_record, blockDim.x);
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb, blockDim.x);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total, blockDim.x);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_lj_virial + atom_i, virial_lj, blockDim.x);
        }
        if (need_du_dlambda)
        {
            Warp_Sum_To(atom_du_dlambda_lj, du_dlambda_lj, blockDim.x);
            if (need_coulomb)
            {
                Warp_Sum_To(atom_du_dlambda_direct, du_dlambda_direct, blockDim.x);
            }
        }
    }
}

void LJ_SOFT_CORE::Initial(CONTROLLER *controller, float cutoff, VECTOR box_length,char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "LJ_soft_core");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    controller[0].printf("START INITIALIZING FEP SOFT CORE FOR LJ AND COULOMB:\n");
    if (controller[0].Command_Exist(this->module_name, "in_file"))
    {
        if (controller[0].Command_Exist("lambda_lj"))
        {
            this->lambda = atof(controller[0].Command("lambda_lj"));
                            controller->printf("    FEP lj lambda: %f\n", this->lambda);
        }
        else
        {
            char error_reason[CHAR_LENGTH_MAX];
            sprintf(error_reason, "Reason:\n\t'lambda_lj' is required for the calculation of LJ_soft_core\n");
            controller->Throw_SPONGE_Error(spongeErrorMissingCommand, "LJ_SOFT_CORE::Initial", error_reason);
        }

        if (controller[0].Command_Exist("soft_core_alpha"))
        {
            this->alpha = atof(controller[0].Command("soft_core_alpha"));
            controller->printf("    FEP soft core alpha: %f\n", this->alpha);
        }
        else
        {
            controller->printf("    FEP soft core alpha is set to default value 0.5\n");
            this->alpha = 0.5;
        }

        if (controller[0].Command_Exist("soft_core_powfer"))
        {
            this->p = atof(controller[0].Command("soft_core_powfer"));
            controller->printf("    FEP soft core powfer: %f\n", this->p);
        }
        else
        {
            controller->printf("    FEP soft core powfer is set to default value 1.0.\n");
            this->p = 1.0;
        }
            
        if (controller[0].Command_Exist("soft_core_sigma"))
        {
            this->sigma = atof(controller[0].Command("soft_core_sigma"));
            controller->printf("    FEP soft core sigma: %f\n", this->sigma);
        }
        else
        {
            controller->printf("    FEP soft core sigma is set to default value 3.0\n");
            this->sigma = 3.0;
        }
        if (controller[0].Command_Exist("soft_core_sigma_min"))
        {
            this->sigma_min = atof(controller[0].Command("soft_core_sigma_min"));
            controller->printf("    FEP soft core sigma min: %f\n", this->sigma_min);
        }
        else
        {
            controller->printf("    FEP soft core sigma min is set to default value 0.0\n");
            this->sigma_min = 0.0;
        }

        FILE *fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"), "r");

        int toscan = fscanf(fp, "%d %d %d", &atom_numbers, &atom_type_numbers_A, &atom_type_numbers_B);
        controller[0].printf("    atom_numbers is %d\n", atom_numbers);
        controller[0].printf("    atom_LJ_type_number_A is %d, atom_LJ_type_number_B is %d\n", atom_type_numbers_A, atom_type_numbers_B);
        pair_type_numbers_A = atom_type_numbers_A * (atom_type_numbers_A + 1) / 2;
        pair_type_numbers_B = atom_type_numbers_B * (atom_type_numbers_B + 1) / 2;
        this->thread_LJ = { 32, 32 };
        LJ_Soft_Core_Malloc();

        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            toscan = fscanf(fp, "%f", h_LJ_AA + i);
            h_LJ_AA[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers_A; i++)
        {
            toscan = fscanf(fp, "%f", h_LJ_AB + i);
            h_LJ_AB[i] *= 6.0f;
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            toscan = fscanf(fp, "%f", h_LJ_BA + i);
            h_LJ_BA[i] *= 12.0f;
        }
        for (int i = 0; i < pair_type_numbers_B; ++i)
        {
            toscan = fscanf(fp, "%f", h_LJ_BB + i);
            h_LJ_BB[i] *= 6.0f;
        }
        for (int i = 0; i < atom_numbers; i++)
        {
            toscan = fscanf(fp, "%d %d", h_atom_LJ_type_A + i, h_atom_LJ_type_B + i);
        }
        fclose(fp);

        if (controller[0].Command_Exist("subsys_division_in_file"))
        {
            FILE * fp = NULL;
            controller->printf("    Start reading subsystem division information:\n");
            Open_File_Safely(&fp, controller[0].Command("subsys_division_in_file"), "r");
            int atom_numbers = 0;
            char lin[CHAR_LENGTH_MAX];
            char * get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
            toscan = sscanf(lin, "%d", &atom_numbers);
            if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers)
            {
                controller->Throw_SPONGE_Error(spongeErrorConflictingCommand, "LJ_SOFT_CORE::Initial",
                    "Reason:\n\t'atom_numbers' (the number of atoms) is diiferent in different input files\n");
            }
            else if (this->atom_numbers == 0)
            {
                this->atom_numbers = atom_numbers;
            }
            for (int i = 0; i < atom_numbers; i++)
            {
                toscan = fscanf(fp, "%d", &h_subsys_division[i]);
            }
            controller->printf("    End reading subsystem division information\n\n");
            fclose(fp);
        }
        else
        {
            controller[0].printf("    subsystem mask is set to 0 as default\n");
            for (int i = 0; i < atom_numbers; i++)
            {
                h_subsys_division[i] = 0;
            }
        }

        Parameter_Host_To_Device();
        is_initialized = 1;
        alpha_lambda_p = alpha * powf(lambda, p);
        alpha_lambda_p_ = alpha * powf(1 - lambda, p);
        sigma_6 = powf(sigma, 6);
        sigma_6_min = powf(sigma_min, 6);
        alpha_lambda_p_1 = alpha * powf(lambda, p-1);
        alpha_lambda_p_1_ = alpha * powf(1.0 - lambda, p-1);
    }
    if (is_initialized)
    {
        this->cutoff = cutoff;
        this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
        Cuda_Malloc_Safely((void **)&uint_crd_with_LJ, sizeof(UINT_VECTOR_LJ_FEP_TYPE)* atom_numbers);
        Copy_LJ_Type_And_Mask_To_New_Crd << <ceilf((float)this->atom_numbers / 32), 32 >> >
            (atom_numbers, uint_crd_with_LJ, d_atom_LJ_type_A, d_atom_LJ_type_B, d_subsys_division);

        controller[0].printf("    Start initializing long range LJ correction\n");
        long_range_factor = 0;
        float *d_factor = NULL;
        Cuda_Malloc_Safely((void**)&d_factor, sizeof(float));
        Reset_List(d_factor, 0.0f, 1, 1);
        Total_C6_Get << < {4, 4}, { 32, 32 } >> >(atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,d_LJ_AB, d_LJ_BB, d_factor, this->lambda);
        cudaMemcpy(&long_range_factor, d_factor, sizeof(float), cudaMemcpyDeviceToHost);
        Reset_List(d_factor, 0.0f, 1, 1);            
        Total_C6_B_A_Get << < {4, 4}, { 32, 32 } >> >(atom_numbers, d_atom_LJ_type_A, d_atom_LJ_type_B,d_LJ_AB, d_LJ_BB, d_factor);
        cudaMemcpy(&long_range_factor_TI, d_factor, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_factor);

        long_range_factor *= -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        long_range_factor_TI *= -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
        this->volume = box_length.x * box_length.y * box_length.z;
        controller[0].printf("        long range correction factor is: %e\n", long_range_factor);
        controller[0].printf("    End initializing long range LJ correction\n");
    }
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller[0].Step_Print_Initial("LJ_soft", "%.2f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n", last_modify_date);
    }
    controller->printf("END INITIALIZING LENNADR JONES SOFT CORE INFORMATION\n\n");
}

void LJ_SOFT_CORE::LJ_Soft_Core_Malloc()
{
    Malloc_Safely((void**)&h_LJ_energy_atom, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_A, sizeof(int)*atom_numbers);
    Malloc_Safely((void**)&h_atom_LJ_type_B, sizeof(int)*atom_numbers);
    Malloc_Safely((void**)&h_LJ_AA, sizeof(float)*pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_AB, sizeof(float)*pair_type_numbers_A);
    Malloc_Safely((void**)&h_LJ_BA, sizeof(float)*pair_type_numbers_B);
    Malloc_Safely((void**)&h_LJ_BB, sizeof(float)*pair_type_numbers_B);
    Malloc_Safely((void**)&h_subsys_division, sizeof(int)*atom_numbers);
    
    Cuda_Malloc_Safely((void**)&d_LJ_energy_sum, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_atom, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_LJ_type_A, sizeof(int)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_atom_LJ_type_B, sizeof(int)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_AA, sizeof(float)*pair_type_numbers_A);
    Cuda_Malloc_Safely((void**)&d_LJ_AB, sizeof(float)*pair_type_numbers_A);
    Cuda_Malloc_Safely((void**)&d_LJ_BA, sizeof(float)*pair_type_numbers_B);
    Cuda_Malloc_Safely((void**)&d_LJ_BB, sizeof(float)*pair_type_numbers_B);
    Cuda_Malloc_Safely((void**)&d_subsys_division, sizeof(int)*atom_numbers);

    Malloc_Safely((void**)&h_LJ_energy_atom_intersys, sizeof(float)*atom_numbers);
    Malloc_Safely((void**)&h_LJ_energy_atom_intrasys, sizeof(float)*atom_numbers);

    Cuda_Malloc_Safely((void**)&d_direct_ene_sum_intersys, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_direct_ene_sum_intrasys, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_sum_intersys, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_sum_intrasys, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_LJ_energy_atom_intersys, sizeof(float)*atom_numbers);
    Cuda_Malloc_Safely((void**)&d_LJ_energy_atom_intrasys, sizeof(float)*atom_numbers);

    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_lj, sizeof(float));
    Malloc_Safely((void**)&h_sigma_of_dH_dlambda_direct, sizeof(float));

    Cuda_Malloc_Safely((void**)&d_sigma_of_dH_dlambda_lj, sizeof(float));
    Cuda_Malloc_Safely((void**)&d_sigma_of_dH_dlambda_direct, sizeof(float));
}

void LJ_SOFT_CORE::Parameter_Host_To_Device()
{
    cudaMemcpy(d_LJ_AB, h_LJ_AB, sizeof(float)*pair_type_numbers_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LJ_AA, h_LJ_AA, sizeof(float)*pair_type_numbers_A, cudaMemcpyHostToDevice);

    cudaMemcpy(d_LJ_BA, h_LJ_BA, sizeof(float)*pair_type_numbers_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LJ_BB, h_LJ_BB, sizeof(float)*pair_type_numbers_B, cudaMemcpyHostToDevice);

    cudaMemcpy(d_atom_LJ_type_A, h_atom_LJ_type_A, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_LJ_type_B, h_atom_LJ_type_B, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_subsys_division, h_subsys_division, sizeof(int)*atom_numbers, cudaMemcpyHostToDevice);
}

void LJ_SOFT_CORE::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;

        free(h_atom_LJ_type_A);
        free(h_atom_LJ_type_B);
        cudaFree(d_atom_LJ_type_A);
        cudaFree(d_atom_LJ_type_B);

        free(h_LJ_AA);
        free(h_LJ_AB);
        free(h_LJ_BA);
        free(h_LJ_BB);
        cudaFree(d_LJ_AA);
        cudaFree(d_LJ_AB);
        cudaFree(d_LJ_BA);
        cudaFree(d_LJ_BB);

        free(h_LJ_energy_atom);
        cudaFree(d_LJ_energy_atom);
        cudaFree(d_LJ_energy_sum);

        cudaFree(uint_crd_with_LJ);

        free(h_subsys_division);
        cudaFree(d_subsys_division);

        h_atom_LJ_type_A = NULL;      
        d_atom_LJ_type_A = NULL;
        h_atom_LJ_type_B = NULL;
        d_atom_LJ_type_B = NULL;       

        h_LJ_AA = NULL;              
        h_LJ_AB = NULL;            
        d_LJ_AA = NULL;          
        d_LJ_AB = NULL;
        
        h_LJ_BA = NULL;
        h_LJ_BB = NULL;
        d_LJ_BA = NULL;
        d_LJ_BB = NULL;

        h_LJ_energy_atom = NULL;    
        d_LJ_energy_atom = NULL;
        d_LJ_energy_sum = NULL;     

        uint_crd_with_LJ = NULL;
        h_subsys_division = NULL;
        d_subsys_division = NULL;
    }
}

void LJ_SOFT_CORE::LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc,
    const ATOM_GROUP *nl, const float pme_beta, const int need_atom_energy, float *atom_energy,
    const int need_virial, float *atom_lj_virial, float *atom_direct_pme_energy)
{
    if (is_initialized)
    {
        Copy_Crd_And_Charge_To_New_Crd << < (this->atom_numbers + 1023) / 1024, 1024 >> >(this->atom_numbers, uint_crd, uint_crd_with_LJ, charge);
        if (atom_numbers == 0)
            return;
        if (!need_atom_energy && !need_virial)
        {
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, false, true, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
            (atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else if (need_atom_energy && !need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, false, true, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
            (atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else if (!need_atom_energy > 0 && need_virial> 0)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, true, true, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, true, true, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
    }
}


void LJ_SOFT_CORE::LJ_Soft_Core_NOPME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
    const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
    const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        Copy_Crd_And_Charge_To_New_Crd << < (this->atom_numbers + 1023) / 1024, 1024 >> > (this->atom_numbers, uint_crd, uint_crd_with_LJ, charge);
        if (atom_numbers == 0)
            return;
        if (!need_atom_energy && !need_virial)
        {
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, false, false, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else if (need_atom_energy && !need_virial)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, false, false, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else if (!need_atom_energy > 0 && need_virial > 0)
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, false, true, false, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else
        {
            cudaMemset(atom_direct_pme_energy, 0, sizeof(float) * this->atom_numbers);
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<true, true, true, false, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy, NULL, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
    }
}

float LJ_SOFT_CORE::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, const float pme_beta, const float* charge, float *pme_direct_ene, int is_download)
{
    if (is_initialized)
    {
        Copy_Crd_And_Charge_To_New_Crd << < (this->atom_numbers + 1023) / 1024, 1024 >> >(atom_numbers, uint_crd, uint_crd_with_LJ, charge);
        Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
        Reset_List << <ceilf((float)atom_numbers / 1024.0f), 1024 >> > (atom_numbers, pme_direct_ene, 0.0f);
        Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, true, false, false, false> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
            (atom_numbers, nl,
                uint_crd_with_LJ, uint_dr_to_dr_cof,
                d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                NULL, pme_beta, d_LJ_energy_atom, NULL, pme_direct_ene, NULL, NULL,
                lambda, alpha, p, sigma_6, sigma_6_min);
        Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
        device_add << <1, 1 >> > (d_LJ_energy_sum, long_range_factor / volume);

        if (is_download)
        {
            cudaMemcpy(&h_LJ_energy_sum, this->d_LJ_energy_sum, sizeof(float), cudaMemcpyDeviceToHost);
            return h_LJ_energy_sum + long_range_factor / this->volume;
        }
        else
        {
            return 0;
        }
    }
    return NAN;
}

float LJ_SOFT_CORE::Get_Partial_H_Partial_Lambda_With_Columb_Direct(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, const ATOM_GROUP *nl,
    const float *charge_B_A, const float pme_beta, const int charge_perturbated, int is_download)
{
    if (is_initialized)
    {
        Copy_Crd_And_Charge_To_New_Crd << < (this->atom_numbers + 1023) / 1024, 1024 >> >(atom_numbers, uint_crd, uint_crd_with_LJ, charge, charge_B_A);

        cudaMemset(d_sigma_of_dH_dlambda_lj, 0, sizeof(float));
        

        if (charge_perturbated > 0)
        {
            cudaMemset(d_sigma_of_dH_dlambda_direct, 0, sizeof(float));
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, false, false, true, true> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    NULL, pme_beta, NULL, NULL, NULL, d_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_direct,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        else
        {
            Lennard_Jones_And_Direct_Coulomb_Soft_Core_CUDA<false, false, false, false, true> << < (atom_numbers + thread_LJ.y - 1) / thread_LJ.y, thread_LJ >> >
                (atom_numbers, nl,
                    uint_crd_with_LJ, uint_dr_to_dr_cof,
                    d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff,
                    NULL, pme_beta, NULL, NULL, NULL, d_sigma_of_dH_dlambda_lj, NULL,
                    lambda, alpha, p, sigma_6, sigma_6_min);
        }
        if (is_download)
        {
            cudaMemcpy(h_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_lj, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sigma_of_dH_dlambda_direct, d_sigma_of_dH_dlambda_direct, sizeof(float), cudaMemcpyDeviceToHost);
            return *h_sigma_of_dH_dlambda_lj + long_range_factor_TI / volume;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return NAN;
    }
}

void LJ_SOFT_CORE::Update_Volume(VECTOR box_length)
{
    if (!is_initialized)
        return;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    this->volume = box_length.x * box_length.y * box_length.z;
}

void LJ_SOFT_CORE::Long_Range_Correction(int need_pressure, float *d_virial, int need_potential, float *d_potential)
{
    if (is_initialized)
    {    
        if (need_pressure > 0)
        {
            device_add << <1, 1 >> >(d_virial, long_range_factor * 6.0f / volume);
        }
        if (need_potential > 0)
        {
            device_add << <1, 1 >> >(d_potential, long_range_factor / volume);
        }
    }
}
