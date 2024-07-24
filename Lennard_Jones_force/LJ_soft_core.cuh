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


#ifndef LJ_SOFT_CORE_CUH
#define LJ_SOFT_CORE_CUH
#include "../common.cuh"
#include "../control.cuh"
#include "Lennard_Jones_force.cuh"

//用于计算LJ_Force时使用的坐标和记录的原子LJ种类序号与原子电荷
#ifndef UINT_VECTOR_LJ_FEP_TYPE_DEFINE
#define UINT_VECTOR_LJ_FEP_TYPE_DEFINE

__device__ __host__ __forceinline__ float Get_Soft_Core_Sigma(const float A, const float B, const float input_sigma_6, const float input_sigma_6_min);
__device__ __host__ __forceinline__ float Get_Soft_Core_Distance(const float A, const float B, const float sigma, const float dr_abs, 
        const float alpha, const float p, const float one_minus_lambda);
__device__ __host__ __forceinline__ float Get_Soft_Core_dU_dlambda(const float F, const float sigma, const float dr_soft_core,
        const float alpha, const float p, const float lambda);
struct UINT_VECTOR_LJ_FEP_TYPE
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
    int LJ_type;
    int LJ_type_B;
    int mask;
    float charge;
    float charge_BA;
    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(UINT_VECTOR_LJ_FEP_TYPE uvec_a, UINT_VECTOR_LJ_FEP_TYPE uvec_b, VECTOR scaler)
    {
        VECTOR dr;
        dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
        dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
        dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
        return dr;
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Energy(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return (0.083333333f * A * dr_6 - 0.166666667f * B) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Force(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float A, const float B)
    {
        return (B - A * powf(dr_abs, -6.0f)) * powf(dr_abs, -8.0f);
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Virial(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return -(B - A * dr_6) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Energy(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float pme_beta)
    {
        return r1.charge * r2.charge * erfcf(pme_beta * dr_abs) / dr_abs;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Force(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float pme_beta)
    {
        float beta_dr = pme_beta * dr_abs;
        return r1.charge * r2.charge * powf(dr_abs, -3.0f) * (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) + erfcf(beta_dr));
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Virial(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, float dr_abs, const float pme_beta)
    {
        float beta_dr = pme_beta * dr_abs;
        return r1.charge * r2.charge / dr_abs * (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) + erfcf(beta_dr));
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_dU_dlambda(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2, const float dr_abs, const float pme_beta)
    {
        return erfcf(pme_beta * dr_abs) * (r2.charge_BA * r1.charge + r2.charge * r1.charge_BA) / dr_abs;
    }
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_Sigma(const float A, const float B, const float input_sigma_6, const float input_sigma_6_min)
    {
        return (A == 0 || B == 0) ? input_sigma_6 : fmaxf(0.5f * A / B, input_sigma_6_min);
    }
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_Distance(const float A, const float B, const float sigma, const float dr_abs, 
        const float alpha, const float p, const float one_minus_lambda)
    {
        float alpha_lambda_p = alpha * powf(one_minus_lambda, p);
        float dr6 = powf(dr_abs, 6.0f);
        return powf(dr6 + alpha_lambda_p * sigma, 0.16666667f);
    }
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_LJ_Force(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2,
        float dr_abs, const float dr_soft_core, const float A, const float B)
    {
        return  powf(dr_abs, 4.0f) *(B - A * powf(dr_soft_core, -6.0f))* powf(dr_soft_core, -12.0f);
    }   
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_LJ_Virial(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2,
        float dr_abs, const float dr_soft_core, const float A, const float B)
    {
        return  -powf(dr_abs, 6.0f) * (B - A * powf(dr_soft_core, -6.0f)) * powf(dr_soft_core, -12.0f);
    }

    friend __device__ __host__ __forceinline__ float Get_Soft_Core_Direct_Coulomb_Force(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2,
        float dr_abs, const float dr_soft_core, const float pme_beta)
    {
        float beta_dr_soft_core = dr_soft_core * pme_beta;
        return r1.charge * r2.charge * powf(dr_abs, 4.0f) * (expf(-beta_dr_soft_core * beta_dr_soft_core) * TWO_DIVIDED_BY_SQRT_PI * pme_beta 
            + erfcf(beta_dr_soft_core) / dr_soft_core) * powf(dr_soft_core, -6.0f);
    }
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_Direct_Coulomb_Virial(UINT_VECTOR_LJ_FEP_TYPE r1, UINT_VECTOR_LJ_FEP_TYPE r2,
        float dr_abs, const float dr_soft_core, const float pme_beta)
    {
        float beta_dr_soft_core = dr_soft_core * pme_beta;
        return r1.charge * r2.charge * powf(dr_abs, 6.0f) * (expf(-beta_dr_soft_core * beta_dr_soft_core) * TWO_DIVIDED_BY_SQRT_PI * pme_beta 
            + erfcf(beta_dr_soft_core) / dr_soft_core) * powf(dr_soft_core, -6.0f);
    }
    friend __device__ __host__ __forceinline__ float Get_Soft_Core_dU_dlambda(const float F, const float sigma, const float dr_soft_core,
        const float alpha, const float p, const float lambda)
    {
        return 0.16666667f * p * alpha * (1 - lambda) * powf(lambda + FLT_MIN, p - 1.0f) * F * powf(dr_soft_core, -4.0f) * sigma;
    }
};
__global__ void Copy_LJ_Type_And_Mask_To_New_Crd(const int atom_numbers, UINT_VECTOR_LJ_FEP_TYPE* new_crd, const int* LJ_type_A, const int* LJ_type_B, const int* mask);
__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_FEP_TYPE *new_crd, const float *charge);
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_FEP_TYPE *new_crd);
#endif

struct LJ_SOFT_CORE
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210730;

    int atom_numbers = 0;           
    int atom_type_numbers_A = 0;
    int atom_type_numbers_B = 0;     
    int pair_type_numbers_A = 0;
    int pair_type_numbers_B = 0;

    int *h_atom_LJ_type_A = NULL;
    int *h_atom_LJ_type_B = NULL;
    int *d_atom_LJ_type_A = NULL;
    int *d_atom_LJ_type_B = NULL;
    
    float *h_LJ_AA = NULL;
    float *h_LJ_AB = NULL;
    float *h_LJ_BA = NULL;
    float *h_LJ_BB = NULL;
    float *d_LJ_AA = NULL;
    float *d_LJ_AB = NULL;
    float *d_LJ_BA = NULL;
    float *d_LJ_BB = NULL;
    
    float *h_LJ_energy_atom = NULL;
    float h_LJ_energy_sum = 0;
    float *d_LJ_energy_atom = NULL;
    float *d_LJ_energy_sum = NULL;

    int * d_subsys_division;
    int * h_subsys_division;

    float *h_LJ_energy_atom_intersys = NULL;
    float *h_LJ_energy_atom_intrasys = NULL;
    float h_LJ_energy_sum_intersys = 0;
    float h_LJ_energy_sum_intrasys = 0;
    float *d_LJ_energy_atom_intersys = NULL;
    float *d_LJ_energy_atom_intrasys = NULL;
    float *d_LJ_energy_sum_intersys = NULL;
    float *d_LJ_energy_sum_intrasys = NULL;

    float *d_direct_ene_sum_intersys = NULL;
    float *d_direct_ene_sum_intrasys = NULL;
    float h_direct_ene_sum = 0.0;
    float h_direct_ene_sum_intersys = 0.0;
    float h_direct_ene_sum_intrasys = 0.0;

    float lambda;
    float alpha;
    float p;
    float alpha_lambda_p;
    float alpha_lambda_p_;
    float alpha_lambda_p_1;
    float alpha_lambda_p_1_;
    float sigma_6;
    float sigma;
    float sigma_min;
    float sigma_6_min;

    float* h_sigma_of_dH_dlambda_lj = NULL;
    float* d_sigma_of_dH_dlambda_lj = NULL;
    
    float* h_sigma_of_dH_dlambda_direct = NULL;
    float* d_sigma_of_dH_dlambda_direct = NULL;

    dim3 thread_LJ = { 32u, 4u };

    float cutoff = 10.0;
    VECTOR uint_dr_to_dr_cof;
    float volume = 0;
    UINT_VECTOR_LJ_FEP_TYPE *uint_crd_with_LJ = NULL;
    float long_range_factor = 0.0;
    float long_range_factor_TI = 0.0;

    void Initial(CONTROLLER *controller, float cutoff, VECTOR box_length, char *module_name = NULL);

    void LJ_Soft_Core_Malloc();

    void Clear();

    void Parameter_Host_To_Device();

    void LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc, const ATOM_GROUP *nl, 
        const float pme_beta, const int need_atom_energy, float *atom_energy, const int need_virial, float *atom_lj_virial, float *atom_direct_pme_energy);

    float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, const float pme_beta, const float * charge, float * direct_ene_sum, int is_download = 1);

    void Update_Volume(VECTOR box_length);

    void Long_Range_Correction(int need_pressure, float *d_virial, int need_potential, float *d_potential);

    float Get_Partial_H_Partial_Lambda_With_Columb_Direct(const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, const ATOM_GROUP* nl,
        const float* charge_B_A, const float pme_beta, const int charge_perturbated = 0, int is_download = 1);

    void LJ_Soft_Core_NOPME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
        const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
        const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy);
};
#endif
