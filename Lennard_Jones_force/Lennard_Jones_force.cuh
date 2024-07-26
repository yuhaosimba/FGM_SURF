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


#ifndef LENNARD_JONES_FORCE_CUH
#define LENNARD_JONES_FORCE_CUH
#include "../common.cuh"
#include "../control.cuh"

//用于计算LJ_Force时使用的坐标和记录的原子LJ种类序号与原子电荷
#ifndef UINT_VECTOR_LJ_TYPE_DEFINE
#define UINT_VECTOR_LJ_TYPE_DEFINE
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446f
__host__ __device__ __forceinline__ int Get_LJ_Type(int a, int b);
__host__ __device__ __forceinline__ int Get_LJ_Type(unsigned int a, unsigned int b);

struct UINT_VECTOR_LJ_TYPE
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
    int LJ_type;
    float charge;
    friend __host__ __device__ __forceinline__ int Get_LJ_Type(int a, int b)
    {
        int y = (b - a);
        int x = y >> 31;
        y = (y ^ x) - x;
        x = b + a;
        int z = (x + y) >> 1;
        x = (x - y) >> 1;
        return (z * (z + 1) >> 1) + x;
    }
    friend __host__ __device__ __forceinline__ int Get_LJ_Type(unsigned int a, unsigned int b)
    {
        int y = (b - a);
        int x = y >> 31;
        y = (y ^ x) - x;
        x = b + a;
        int z = (x + y) >> 1;
        x = (x - y) >> 1;
        return (z * (z + 1) >> 1) + x;
    }
    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(UINT_VECTOR_LJ_TYPE uvec_a, UINT_VECTOR_LJ_TYPE uvec_b, VECTOR scaler)
    {
        VECTOR dr;
        dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
        dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
        dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
        return dr;
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Energy(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return (0.083333333f * A * dr_6 - 0.166666667f * B) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Force(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float A, const float B)
    {
        return (B - A * powf(dr_abs, -6.0f)) * powf(dr_abs, -8.0f);
    }
    friend __device__ __host__ __forceinline__ float Get_LJ_Virial(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float A, const float B)
    {
        float dr_6 = powf(dr_abs, -6.0f);
        return -(B - A * dr_6) * dr_6;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Energy(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float pme_beta)
    {
        return r1.charge * r2.charge * erfcf(pme_beta * dr_abs) / dr_abs;
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Force(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float pme_beta)
    {
        float beta_dr = pme_beta * dr_abs;
        return r1.charge * r2.charge * powf(dr_abs, -3.0f) * (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) + erfcf(beta_dr));
    }
    friend __device__ __host__ __forceinline__ float Get_Direct_Coulomb_Virial(UINT_VECTOR_LJ_TYPE r1, UINT_VECTOR_LJ_TYPE r2, float dr_abs, const float pme_beta)
    {
        float beta_dr = pme_beta * dr_abs;
        return r1.charge * r2.charge / dr_abs * (beta_dr * TWO_DIVIDED_BY_SQRT_PI * expf(-beta_dr * beta_dr) + erfcf(beta_dr));
    }
};
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers, UINT_VECTOR_LJ_TYPE *new_crd, const int *LJ_type);
__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd, const float *charge);
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd);
#endif

//用于记录与计算LJ相关的信息
struct LENNARD_JONES_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210830;

    //a = LJ_A between atom[i] and atom[j]
    //b = LJ_B between atom[i] and atom[j]
    //E_lj = a/12 * r^-12 - b/6 * r^-6;
    //F_lj = (a * r^-14 - b * r ^ -6) * dr
    int atom_numbers = 0;           //原子数
    int atom_type_numbers = 0;      //原子种类数
    int pair_type_numbers = 0;      //原子对种类数
    

    int *h_atom_LJ_type = NULL;        //原子对应的LJ种类
    int *d_atom_LJ_type = NULL;        //原子对应的LJ种类
    
    float *h_LJ_A = NULL;              //LJ的A系数
    float *h_LJ_B = NULL;              //LJ的B系数
    float *d_LJ_A = NULL;              //LJ的A系数
    float *d_LJ_B = NULL;              //LJ的B系数
    
    float *h_LJ_energy_atom = NULL;    //每个原子的LJ的能量
    float h_LJ_energy_sum = 0;     //所有原子的LJ能量和
    float *d_LJ_energy_atom = NULL;    //每个原子的LJ的能量
    float *d_LJ_energy_sum = NULL;     //所有原子的LJ能量和

    dim3 thread_LJ = { 32, 32 }; // cuda参数
    //初始化
    void Initial(CONTROLLER *controller, float cutoff, VECTOR box_length, const char *module_name = NULL);
    //从amber的parm文件里读取
    void Initial_From_AMBER_Parm(const char *file_name, CONTROLLER controller);
    //清除内存
    void Clear();
    //分配内存
    void LJ_Malloc();
    //参数传到GPU上
    void Parameter_Host_To_Device();
    

    float cutoff = 10.0;
    VECTOR uint_dr_to_dr_cof;
    float volume = 0;
    UINT_VECTOR_LJ_TYPE *uint_crd_with_LJ = NULL;

    
    //可以根据外界传入的need_atom_energy和need_virial，选择性计算能量和维里。其中的维里对PME直接部分计算的原子能量，在和PME其他部分加和后即维里。
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc,
        const ATOM_GROUP *nl, const float pme_beta, const int need_atom_energy, float *atom_energy,
        const int need_virial, float *atom_lj_virial, float *atom_direct_pme_energy);

    //长程能量和维里修正
    float long_range_factor = 0;
    //求力的时候对能量和维里的长程修正
    void Long_Range_Correction(int need_pressure, float *d_virial, int need_potential, float *d_potential);

    //获得能量
    float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, const float pme_beta, const float *charge, float *pme_direct_energy, int is_download = 1);
    //获得能量，屏蔽PME
    float Get_Energy(const UNSIGNED_INT_VECTOR* uint_crd, const ATOM_GROUP* nl, int is_download);
    //更新体积
    void Update_Volume(VECTOR box_length);
    
    void LJ_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
        const ATOM_GROUP* nl, const float pme_beta, const int need_atom_energy, float* atom_energy,
        const int need_virial, float* atom_lj_virial, float* atom_direct_pme_energy);

    //能量从GPU到CPU
    void Energy_Device_To_Host();

    void LJ_NOPME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR* uint_crd, const float* charge, VECTOR* frc,
        const ATOM_GROUP* nl,  const int need_atom_energy, float* atom_energy,
        const int need_virial, float* atom_lj_virial);
};
#endif //LENNARD_JONES_FORCE_CUH(Lennard_Jones_force.cuh)
