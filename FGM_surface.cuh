#ifndef FGM_SURF_CUH
#define FGM_SURF_CUH
#include "common.cuh"
#include "control.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h> 
#include <fstream>
#include <iostream>
#include <cublas_v2.h>
#include<time.h>
#include <math.h>
#include "cstdlib"
#include"curand_kernel.h"
#include "sm_60_atomic_functions.h"



struct FGM_SURF {
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20240722;


    // 存储有限元网格基本信息
    float box_x, box_y, box_z;
    float dx, dy, dz;
    int nx, ny, nz;
    int first_cut_sgn;


    // 存储电荷数目、电荷原子序号
    int n_charge; 
    int N_start;   // 第一个带电荷原子的序号
    int* h_charge_sign;
    int* d_charge_sign;


    // 初始化cublas、cusparse信息
    float     alpha = 0.0;
    float     beta = 0.0;
    float     r_r = 0.0;
    float     r_r_next = 0.0;
    float     p_A_p = 0.0;
    float     err = 0.0;
    const float     one = 1.0;
    const float     zero = 0.0;
    const float     minus = -1.0;
    float* d_r;
    float* d_p, * d_Ap;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vec_phi, vec_r, vec_p, vec_Ap;
    cusparseSpMatDescr_t mat_Trans;
    cusparseDnVecDescr_t vec_phi_ortho;
    cusparseDnVecDescr_t vec_charge, vec_charge_ortho;
    void* dBuffer = NULL;
    void* dBuffer2 = NULL;
    size_t bufferSize = 0;
    size_t bufferSize2 = 0;
    cublasHandle_t bandle = NULL;
    cusparseHandle_t handle = NULL;


    // 初始化电势矩阵
    // CSR电势求解矩阵参数
    int A_num_rows;  // dim of dense matrix
    int A_nnz;       // non-zero 
    // Host 电势求解矩阵相关信息
    int* hA_csrOffsets, * hA_columns;
    float* hA_values, * h_const;
    float* h_phi;  // 初猜电势
    // Device 电势求解矩阵相关信息
    int* dA_csrOffsets, * dA_columns;
    float* dA_values, * d_const;
    float* d_phi;
    // Transition matrix from calc to NX NY NZ 计算 <->正交向量化坐标相互转换
    int* h_from_calc_to_ortho;
    int* h_from_ortho_to_calc;
    int* d_from_calc_to_ortho;
    int* d_from_ortho_to_calc;
    // 电势的正交化网格信息
    float* h_phi_ortho;    // data-shape: x*(NY*NZ) + y*NZ + z
    float* d_phi_ortho;
    // 电荷离散化信息
    float* d_charge, * d_charge_ortho; // 网格电荷 &  正交网格电荷
    float* result_phi;


    // 等势球面信息
    // 492个格点的单位球面。 球面格点数，边数，面数
    const int N_points = 492;
    int N_edges, N_faces;
    //存储球面网络位置、初始边长、加速度、速度
    float* h_points, * h_edges_length_unit, * d_points, * d_edges_length_unit, * d_Acc, * d_Vel;;
    int* h_edges, * h_faces, * d_edges, * d_faces; // 存储边与面的信息
    float* shape_after_force; // 存储球面受力后的形状
    // Debug: Set parameters
    // VDW_boundary 是等势面半径，R 是采样球面平衡半径
    float k1 = 100000.0, VDW_boundary = 20.3, Attach_force = 1000., k2 = 1000.0, k3 = 5000.0, k4 = 30000., k5 = 30000, R = 5.0, dt_shpere = 0.0001;
    // 摩擦系数
    float gamma = 0.1;
    const int max_iter = 500;
    float Err = 0.;

    // 等势面形状
    int type_eq_shape = 1;  //默认为球面
    // 等势边界球的中心 
    int N_eq;
    float* h_eq_surface_center;
    float* d_eq_surface_center;


    // 静电力计算相关信息
    float* d_sphere_phi_each_point;             // 每个采样点所受的电势
    float* d_sphere_Electric_Field_each_point;  // 每个采样点所受的电场力
    float* d_E_result;  // 最终电场力结果
    float* h_E_result;  // 最终电场力结果
    float* d_solid_angle;  // 存储立体角向量
    float* d_vec_one;    // 全1向量，用于求和

    //初始化模块
    void Initial(CONTROLLER* controller, const int atom_numbers, const char* module_name = NULL);
    //内存分配
    void Memory_Allocate_MESH();
    void Memory_Allocate_SPHERE();
    //清空模块
    void Clear_MESH();
    void Clear_SPHERE();
    //拷贝cpu中的参数到gpu
    void Parameter_Host_To_Device_MESH();
    void Parameter_Host_To_Device_SPHERE();
    void Potential_file_saver(std::string file_name);
    void Sphere_saver(std::string file_name);
};

extern __global__ void refresh_to_zero(float* d_phi_ortho, int n);
extern __global__ void multiply_by_constant(float* d_array, int n, float cst);
extern __global__ void add_by_constant(float* d_array, int n, float cst);
extern __global__ void add_by_vector(float* d_array, int N_points, float* d_charge_center, int chg);
extern __global__ void add_by_vector(float* d_array, int N_points, VECTOR* d_charge_center, int chg);
extern __global__ void plus_two_vectors(float* array_1, float* array_2, int n);


// 从正交网格转换到计算网格
extern __global__ void trans_from_ortho_to_calc(float* d_phi_ortho, float* d_phi, int* d_from_ortho_to_calc, int nx, int ny, int nz);
// 从计算网格转换到正交网格
extern __global__ void trans_from_calc_to_ortho(float* d_phi, float* d_phi_ortho, int* d_from_calc_to_ortho, int first_cut_sgn);



// FGM计算静电力，只计算力，不计算能量与维里
void FGM_CalC_Force_Only_with_Exclude(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge,
    const int* excluded_list_start, const int* excluded_list, const int* excluded_numbers, VECTOR* frc);

/*--------------------------------------------------------电势求解函数-------------------------------------------------------------*/

// 网格电荷插值函数(kernel)
__global__ void charge_interpolation_cuda(VECTOR* crd, float* charge_q, int N_start, int nx, int ny, int nz, float box_x, float box_y, float box_z,
    float dx, float dy, float dz, int n_charge, float* d_charge_ortho);

//  网格电荷插值函数
void Charge_Interpolation(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge);

// 共轭梯度法方程组求解器
void CG_Solver(FGM_SURF fgm_surf);


/*--------------------------------------------------------格林函数曲面积分函数-------------------------------------------------------------*/
__global__ void boundary_force_shpere(int N_points, float* points, int N_eq, float* boundary_atom_center,
    const float k1, const float VDW_boundary, const float Attach_force, float* Acc);


__global__ void triangle_force(float* points, int* faces, const float k2, float* Acc, int N_faces);

__global__ void edge_force(float* points, int* edges, float* edge_length_init,
    const float k3, const float k4, float* Acc, int N_edges, float R);


__global__ void center_force(int N_points, float* points, VECTOR* crd,
    const float k5, const float R, float* Acc, int chg);   // chg 是电荷的原子编号

// 更新位置，方式采用Leap-Frog算法
__global__ void update_position_Leapfrog(int N_points, float* points, float* Vel, const float dt);
// 更新速度，方式采用Leap-Frog算法
__global__ void update_velocity_Leapfrog(int N_points, float* Vel, float* Acc, const float dt);

// 自适应曲面迭代生成器
void Sphere_Autoconfort_Iterator(FGM_SURF fgm_surf, VECTOR* crd, int chg);

/*-------------------------------------------------基于格林曲面积分的静电力计算方法--------------------------------------------------------*/


// 计算球面的电势/电场分布
void Eletric_Field_Calculator(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge, int chg, const int* excluded_list_start, const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc);

// 计算球面的电势/电场分布(kernel)
__global__ void Potential_and_Elecfield_calculator(float* phi, int* from_ortho_to_calc,
    int N_points, float* sphere_crd,
    int nx, int ny, int nz,
    float box_x, float box_y, float box_z,
    float* sphere_phi_each_point, float* sphere_Electric_Field_each_point);

// 计算格林曲面积分对静电力的贡献
__global__ void Calc_Green_Integral(int N_faces, int* faces, float* sphere_crd,
    int chg, VECTOR* crd, float* sphere_phi_each_point, float* sphere_Electric_Field_each_point, float* E_result, int N_start);

// 计算空间中某点相对于格林曲面的位置，方法是计算Winding-Number
__global__ void Calculate_Solid_Angle_in_current_Hull(int N_faces, int* faces, float* sphere_crd,
    int chg, int chg_other, VECTOR* crd, float* d_solid_angle, float box_x, float box_y, float box_z, int N_start);

// 计算格林曲面内电荷的直接作用
__global__ void Calc_charge_inside(int chg, int chg_other, VECTOR* crd, float* d_charge, float* E_result, float box_x, float box_y, float box_z, int N_start);

// 给定电荷处的电场计算静电力，传入frc中
__global__ void Calc_FGM_Force_by_Elec_field(const int n_charge, const int N_start, float* d_charge, VECTOR* frc, float* E_result);




#endif