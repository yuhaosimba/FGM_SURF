#include "solvent_LJ.cuh"

__global__ void Uint_Vector_Soft_Core_To_Hard_Core(int atom_numbers, UINT_VECTOR_LJ_TYPE* hard_core_ucrd, const UINT_VECTOR_LJ_FEP_TYPE* soft_core_ucrd)
{
    int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
    if (atom_i < atom_numbers)
    {
        UINT_VECTOR_LJ_TYPE new_crd;
        UINT_VECTOR_LJ_FEP_TYPE crd = soft_core_ucrd[atom_i];
        new_crd.uint_x = crd.uint_x;
        new_crd.uint_y = crd.uint_y;
        new_crd.uint_z = crd.uint_z;
        new_crd.charge = crd.charge;
        new_crd.LJ_type = crd.LJ_type;
        hard_core_ucrd[atom_i] = new_crd;
    }
}

template<int WAT_POINTS, bool need_force, bool need_energy, bool need_virial, bool need_coulomb>
static __global__ void Lennard_Jones_And_Direct_Coulomb_CUDA(
    const int atom_numbers, const ATOM_GROUP* nl, const int solvent_start_residue,
    const int res_numbers, const int* d_res_start, const int* d_res_end,
    const UINT_VECTOR_LJ_TYPE* uint_crd, const VECTOR boxlength,
    const float* LJ_type_A, const float* LJ_type_B, const float cutoff,
    VECTOR* frc, const float pme_beta, float* atom_energy, float* atom_lj_virial, float* atom_direct_cf_energy)
{
    int residue_i = blockDim.y * blockIdx.x + threadIdx.y + solvent_start_residue;
    __shared__ UINT_VECTOR_LJ_TYPE r1s[128];
    if (residue_i < res_numbers)
    {
        VECTOR frc_record[4] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
        VECTOR frc_record_j;
        int atom_i = d_res_start[residue_i];
        if (threadIdx.x < WAT_POINTS)
        {
            r1s[threadIdx.y * WAT_POINTS + threadIdx.x] = uint_crd[atom_i + threadIdx.x];
        }
        __syncwarp();
        ATOM_GROUP nl_i = nl[atom_i];
        float virial_lj = 0.;
        float energy_total = 0.;
        float energy_coulomb = 0.;
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
        {
            int atom_j = nl_i.atom_serial[j];
            UINT_VECTOR_LJ_TYPE r2 = uint_crd[atom_j];
            frc_record_j = { 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < WAT_POINTS; i++)
            {
                UINT_VECTOR_LJ_TYPE r1 = r1s[threadIdx.y * WAT_POINTS + i];
                VECTOR dr = Get_Periodic_Displacement(r2, r1, boxlength);
                float dr_abs = norm3df(dr.x, dr.y, dr.z);
                if (dr_abs < cutoff)
                {
                    int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                    float A = LJ_type_A[atom_pair_LJ_type];
                    float B = LJ_type_B[atom_pair_LJ_type];
                    if (need_force)
                    {
                        float frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                        if (need_coulomb)
                        {
                            float frc_cf_abs = Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                            frc_abs = frc_abs - frc_cf_abs;
                        }
                        VECTOR frc_lin = frc_abs * dr;
                        frc_record[i] = frc_record[i] + frc_lin;
                        frc_record_j = frc_record_j - frc_lin;
                    }
                    if (need_coulomb && (need_energy || need_virial))
                    {
                        energy_coulomb += Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_energy)
                    {
                        energy_total += Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    }
                    if (need_virial)
                    {
                        virial_lj += Get_LJ_Virial(r1, r2, dr_abs, A, B);
                    }
                }
            }
            if (need_force)
            {
                atomicAdd(frc + atom_j, frc_record_j);
            }
        }
        if (need_force)
        {
            for (int i = 0; i < WAT_POINTS; i++)
            {
                Warp_Sum_To(frc + atom_i + i, frc_record[i]);
            }
        }
        if (need_coulomb && (need_energy || need_virial))
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, energy_coulomb);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, energy_total);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_lj_virial + atom_i, virial_lj);
        }
    }
}

void SOLVENT_LENNARD_JONES::Initial(CONTROLLER* controller, LENNARD_JONES_INFORMATION* lj, LJ_SOFT_CORE *lj_soft, 
    int res_num, int* res_start, int* res_end, bool default_enable, const char *module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "solvent_LJ");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    bool enable;
    if (!controller->Command_Exist(this->module_name))
    {
        enable = default_enable && res_num > 10;
    }
    else
    {
        enable = controller->Get_Bool(this->module_name, "SOLVENT_LENNARD_JONES::Initial");
    }
    water_points = res_end[res_num - 1] - res_start[res_num - 1];
    if (water_points != 3 && water_points != 4)
    {
        enable = false;
    }
    if (enable)
    {
        controller->printf("START INITIALIZING SOLVENT LJ:\n");
        this->lj_info = lj;
        this->lj_soft_info = lj_soft;
        solvent_numbers = 0;
        solvent_start = res_num;
        for (int i = res_num - 1; i >= 0; i -= 1)
        {
            int res_atom_numbers = res_end[i] - res_start[i];
            if (res_atom_numbers == water_points)
            {
                solvent_numbers += res_atom_numbers;
                solvent_start -= 1;
            }
        }
        controller->printf("    the number of solvent atoms is %d (started from Residue #%d)\n", solvent_numbers, solvent_start);
        if (solvent_numbers > 0)
        {
            is_initialized = 1;
            if (lj_soft_info->is_initialized)
            {
                Cuda_Malloc_Safely((void**)&soft_to_hard_crd, sizeof(UINT_VECTOR_LJ_TYPE) * lj_soft_info->atom_numbers);
            }
        }
    }
    if (!is_initialized)
    {
        solvent_numbers = 0;
        controller->printf("SOLVENT LJ IS NOT INITIALIZED\n\n");
    }
    else if (!is_controller_printf_initialized)
    {
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n", last_modify_date);
        controller->printf("END INITIALIZING SOLVENT LJ\n\n");
    }
    else
    {
        controller->printf("END INITIALIZING SOLVENT LJ\n\n");
    }
}

#define LJ_Direct_Coulomb_CUDA(wat_points, need_force, need_energy, need_virial, need_coulomb)  \
Lennard_Jones_And_Direct_Coulomb_CUDA<wat_points, need_force, need_energy, need_virial, need_coulomb> << < block_LJ, thread_LJ >> > \
(atom_numbers, nl, solvent_start, residue_numbers, d_res_start, d_res_end, \
    lj_info->uint_crd_with_LJ, lj_info->uint_dr_to_dr_cof, \
    lj_info->d_LJ_A, lj_info->d_LJ_B, lj_info->cutoff, \
    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy)

#define Soft_LJ_Direct_Coulomb_CUDA(wat_points, need_force, need_energy, need_virial, need_coulomb)  \
Lennard_Jones_And_Direct_Coulomb_CUDA<wat_points, need_force, need_energy, need_virial, need_coulomb> << < block_LJ, thread_LJ >> > \
(atom_numbers, nl, solvent_start, residue_numbers, d_res_start, d_res_end, \
    soft_to_hard_crd, lj_soft_info->uint_dr_to_dr_cof, \
    lj_soft_info->d_LJ_AA, lj_soft_info->d_LJ_AB, lj_soft_info->cutoff, \
    frc, pme_beta, atom_energy, atom_lj_virial, atom_direct_pme_energy);

void SOLVENT_LENNARD_JONES::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const int residue_numbers, const int* d_res_start, const int* d_res_end, const UNSIGNED_INT_VECTOR* uint_crd, 
    const float* charge, VECTOR* frc, const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy, 
    const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        dim3 thread_LJ = { 32u, 32u };
        dim3 block_LJ = (residue_numbers - solvent_start + thread_LJ.y - 1) / thread_LJ.y;
        switch (water_points)
        {
        case 3:
            if (lj_info->is_initialized)
            {
                if (!need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, false, false, true);
                }
                else if (need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, true, false, true);
                }
                else if (!need_atom_energy && need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, false, true, true);
                }
                else
                {
                    LJ_Direct_Coulomb_CUDA(3, true, true, true, true);
                }
            }
            else if (lj_soft_info->is_initialized)
            {
                Uint_Vector_Soft_Core_To_Hard_Core << <(atom_numbers + 1023) / 1024, 1024 >> > (atom_numbers, soft_to_hard_crd, lj_soft_info->uint_crd_with_LJ);
                if (!need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, false, false, true);
                }
                else if (need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, true, false, true);
                }
                else if (!need_atom_energy && need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, false, true, true);
                }
                else
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, true, true, true);
                }
            }
            break;
        case 4:
            if (lj_info->is_initialized)
            {
                if (!need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, false, false, true);
                }
                else if (need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, true, false, true);
                }
                else if (!need_atom_energy && need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, false, true, true);
                }
                else
                {
                    LJ_Direct_Coulomb_CUDA(4, true, true, true, true);
                }
            }
            else if (lj_soft_info->is_initialized)
            {
                Uint_Vector_Soft_Core_To_Hard_Core << <(atom_numbers + 1023) / 1024, 1024 >> > (atom_numbers, soft_to_hard_crd, lj_soft_info->uint_crd_with_LJ);
                if (!need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, false, false, true);
                }
                else if (need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, true, false, true);
                }
                else if (!need_atom_energy && need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, false, true, true);
                }
                else
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, true, true, true);
                }
            }
            break;
        }
    }
}



void SOLVENT_LENNARD_JONES::LJ_NOPME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const int residue_numbers, const int* d_res_start, const int* d_res_end, const UNSIGNED_INT_VECTOR* uint_crd,
    const float* charge, VECTOR* frc, const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
    const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy)
{
    if (is_initialized)
    {
        dim3 thread_LJ = { 32u, 32u };
        dim3 block_LJ = (residue_numbers - solvent_start + thread_LJ.y - 1) / thread_LJ.y;
        switch (water_points)
        {
        case 3:
            if (lj_info->is_initialized)
            {
                if (!need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, false, false, false);
                }
                else if (need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, true, false, false);
                }
                else if (!need_atom_energy && need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(3, true, false, true, false);
                }
                else
                {
                    LJ_Direct_Coulomb_CUDA(3, true, true, true, false);
                }
            }
            else if (lj_soft_info->is_initialized)
            {
                Uint_Vector_Soft_Core_To_Hard_Core << <(atom_numbers + 1023) / 1024, 1024 >> > (atom_numbers, soft_to_hard_crd, lj_soft_info->uint_crd_with_LJ);
                if (!need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, false, false, false);
                }
                else if (need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, true, false, false);
                }
                else if (!need_atom_energy && need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, false, true, false);
                }
                else
                {
                    Soft_LJ_Direct_Coulomb_CUDA(3, true, true, true, false);
                }
            }
            break;
        case 4:
            if (lj_info->is_initialized)
            {
                if (!need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, false, false, false);
                }
                else if (need_atom_energy && !need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, true, false, false);
                }
                else if (!need_atom_energy && need_virial)
                {
                    LJ_Direct_Coulomb_CUDA(4, true, false, true, false);
                }
                else
                {
                    LJ_Direct_Coulomb_CUDA(4, true, true, true, false);
                }
            }
            else if (lj_soft_info->is_initialized)
            {
                Uint_Vector_Soft_Core_To_Hard_Core << <(atom_numbers + 1023) / 1024, 1024 >> > (atom_numbers, soft_to_hard_crd, lj_soft_info->uint_crd_with_LJ);
                if (!need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, false, false, false);
                }
                else if (need_atom_energy && !need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, true, false, false);
                }
                else if (!need_atom_energy && need_virial)
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, false, true, false);
                }
                else
                {
                    Soft_LJ_Direct_Coulomb_CUDA(4, true, true, true, false);
                }
            }
            break;
        }
    }
}
