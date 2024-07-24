#include "FGM_surface.cuh"


void FGM_SURF::Initial(CONTROLLER* controller, const int atom_numbers, const char* module_name)
{
    //给予bond模块一个默认名字：fgm_surf
    if (module_name == NULL)
    {
        strcpy(this->module_name, "fgm_surf");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    //指定读入bond信息的后缀，默认为in_file，即合成为fgm_surf_in_file
    char file_name_suffix_1[CHAR_LENGTH_MAX];   
    sprintf(file_name_suffix_1, "mesh_in_file"); 
    if (controller[0].Command_Exist(this->module_name, file_name_suffix_1))
    {
        controller[0].printf("START INITIALIZING FGM MESH INFO (%s_%s):\n", this->module_name, file_name_suffix_1);
        FILE* fp = NULL;
        Open_File_Safely(&fp, controller[0].Command(this->module_name, file_name_suffix_1), "r");
        fscanf(fp, "%d %d %d %d %d", &A_num_rows, &A_nnz, &nx, &ny, &nz);
        fscanf(fp, "%f %f %f %d", &box_x, &box_y, &box_z, &first_cut_sgn);
        dx = box_x / nx; dy = box_y / ny; dz = box_z / nz;
        Memory_Allocate_MESH();
        for (int i = 0; i < A_nnz; i++) { fscanf(fp, "%d", &this->hA_columns[i]); }
        for (int i = 0; i < A_num_rows + 1; i++) { fscanf(fp, "%d", &this->hA_csrOffsets[i]); }
        for (int i = 0; i < A_nnz; i++) { fscanf(fp, "%f", &this->hA_values[i]); }
        for (int i = 0; i < A_num_rows; i++) { fscanf(fp, "%f", &this->h_const[i]); }
        for (int i = 0; i < A_num_rows; i++) { fscanf(fp, "%f", &this->h_phi[i]); }
        for (int i = 0; i < nx * ny * nz; i++) { fscanf(fp, "%d",&this->h_from_ortho_to_calc[i]); }
        for (int i = 0; i < first_cut_sgn; i++) { fscanf(fp, "%d", &this->h_from_calc_to_ortho[i]); }
        fclose(fp);
        Parameter_Host_To_Device_MESH();
        controller[0].printf("END INITIALIZING FGM MESH\n \n");
    }
    
    char charge_start[CHAR_LENGTH_MAX];   // 带电荷原子起始序号信息
    sprintf(charge_start, "charge_start");
    if (controller[0].Command_Exist(charge_start)) 
    {
        N_start = atoi(controller[0].Command(charge_start));
        controller[0].printf("FIRST CHARGE'S SIGN = %d \n \n", N_start);
        n_charge = atom_numbers - N_start;
        this->h_charge_sign = (int*)malloc(this->n_charge * sizeof(int));
        for (int i = 0; i < n_charge; i++) { this->h_charge_sign[i] = N_start + i; }
        cudaMalloc((void**)&this->d_charge_sign, this->n_charge * sizeof(int));
        cudaMemcpy(this->d_charge_sign, this->h_charge_sign, this->n_charge * sizeof(int), cudaMemcpyHostToDevice);
    }
   
    char eq_shape[CHAR_LENGTH_MAX];   // 等势面形状信息
    sprintf(eq_shape, "eq_surf_shape_in_file");
    if (controller[0].Command_Exist(eq_shape))
    {
        controller[0].printf("START INITIALIZING EQ_SURFACE SHAPE INFO (%s_%s):\n", this->module_name, eq_shape);
        FILE* fp3 = NULL;
        Open_File_Safely(&fp3, controller[0].Command(eq_shape), "r");
        fscanf(fp3, "%d", &this->type_eq_shape);
        if (this->type_eq_shape == 1) {
            fscanf(fp3, "%d", &this->N_eq);
            h_eq_surface_center = (float*)malloc(this->N_eq * 3 * sizeof(float));
            for (int i = 0; i < this->N_eq; i++) {fscanf(fp3, "%f %f %f %f", &this->h_eq_surface_center[3 * i], &this->h_eq_surface_center[3 * i + 1], &this->h_eq_surface_center[3 * i + 2], &this->VDW_boundary);}
            printf("    N_eq = %d, center = %f, %f, %f\n", N_eq, h_eq_surface_center[0], h_eq_surface_center[1], h_eq_surface_center[2]);
        }
    }


    char file_name_suffix_2[CHAR_LENGTH_MAX];
    sprintf(file_name_suffix_2, "sphere_in_file");
    if (controller[0].Command_Exist(this->module_name, file_name_suffix_2))
    {
        controller[0].printf("START INITIALIZING FGM GREEN SPHERE INFO (%s_%s):\n", this->module_name, file_name_suffix_2);
        FILE* fp2 = NULL;
        Open_File_Safely(&fp2, controller[0].Command(this->module_name, file_name_suffix_2), "r");
        fscanf(fp2, "%d", &N_edges);
        fscanf(fp2, "%d", &N_faces);
        Memory_Allocate_SPHERE();
        for (int i = 0; i < N_points; i++) { fscanf(fp2, "%f %f %f", &h_points[3 * i], &h_points[3 * i + 1], &h_points[3 * i + 2]); }
        for (int i = 0; i < N_edges; i++) { fscanf(fp2, "%f", &h_edges_length_unit[i]); }
        for (int i = 0; i < N_edges; i++) { fscanf(fp2, "%d %d", &h_edges[2 * i], &h_edges[2 * i + 1]); }
        for (int i = 0; i < N_faces; i++) { fscanf(fp2, "%d %d %d", &h_faces[3 * i], &h_faces[3 * i + 1], &h_faces[3 * i + 2]); }
        Parameter_Host_To_Device_SPHERE();
        controller[0].printf("END INITIALIZING FGM SPHERE\n \n");
        fclose(fp2);
    }

    is_initialized = 1;
}

void FGM_SURF::Memory_Allocate_MESH()
{
    if (!Malloc_Safely((void**)&(this->hA_csrOffsets), sizeof(int) *(this->A_num_rows + 1)))
        printf("        Error occurs when malloc FGM_SURF::hA_csrOffsets in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->hA_columns), sizeof(int) * this->A_nnz))
        printf("        Error occurs when malloc FGM_SURF::hA_columns in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->hA_values), sizeof(float) * this->A_nnz))
        printf("        Error occurs when malloc FGM_SURF::hA_values in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->h_phi), sizeof(float) * this->A_num_rows))
        printf("        Error occurs when malloc FGM_SURF::h_phi in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->h_const), sizeof(float) * this->A_num_rows))
        printf("        Error occurs when malloc FGM_SURF::h_const in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->h_from_calc_to_ortho), sizeof(int) * this->first_cut_sgn))
        printf("        Error occurs when malloc FGM_SURF::h_from_calc_to_ortho in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->h_from_ortho_to_calc), sizeof(int) * this->nx * this->ny * this->nz))
        printf("        Error occurs when malloc FGM_SURF::h_from_ortho_to_calc in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->h_phi_ortho), sizeof(float) * this->nx * this->ny * this->nz))
        printf("        Error occurs when malloc FGM_SURF::h_phi_ortho in FGM_SURF::Memory_Allocate_MESH");
    if (!Malloc_Safely((void**)&(this->result_phi), sizeof(float) * this->A_num_rows))
        printf("        Error occurs when malloc FGM_SURF::result_phi in FGM_SURF::Memory_Allocate_MESH");

    if (!Cuda_Malloc_Safely((void**)&this->dA_csrOffsets, sizeof(int) * (this->A_num_rows + 1)))
        printf("        Error occurs when CUDA malloc FGM_SURF::dA_csrOffsets in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->dA_columns, sizeof(int) * this->A_nnz))
        printf("        Error occurs when CUDA malloc FGM_SURF::dA_columns in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->dA_values, sizeof(float) * this->A_nnz))
        printf("        Error occurs when CUDA malloc FGM_SURF::dA_values in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_phi, sizeof(float) * this->A_num_rows))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_phi in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_const, sizeof(float) * this->A_num_rows))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_const in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_phi_ortho, sizeof(float) * this->nx * this->ny * this->nz)) // 正交网格电势
        printf("        Error occurs when CUDA malloc FGM_SURF::d_phi_ortho in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_charge, sizeof(float) * this->A_num_rows))  // 网格电荷
        printf("        Error occurs when CUDA malloc FGM_SURF::d_charge in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_charge_ortho, sizeof(float) * this->nx * this->ny * this->nz)) // 正交网格电荷
        printf("        Error occurs when CUDA malloc FGM_SURF::d_charge_ortho in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_r, sizeof(float) * this->A_num_rows))  
        printf("        Error occurs when CUDA malloc FGM_SURF::d_r in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_p, sizeof(float) * this->A_num_rows))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_p in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_Ap, sizeof(float) * this->A_num_rows))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_Ap in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_from_calc_to_ortho, sizeof(int) * this->first_cut_sgn))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_from_calc_to_ortho in FGM_SURF::Memory_Allocate_MESH");
    if (!Cuda_Malloc_Safely((void**)&this->d_from_ortho_to_calc, sizeof(int) * this->nx * this->ny * this->nz))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_from_ortho_to_calc in FGM_SURF::Memory_Allocate_MESH");
}

void FGM_SURF::Parameter_Host_To_Device_MESH() {
    
    cudaMemcpy(this->dA_csrOffsets, this->hA_csrOffsets, (this->A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->dA_columns, this->hA_columns, this->A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->dA_values, this->hA_values, this->A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_phi, this->h_phi, this->A_num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_const, this->h_const, this->A_num_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_from_calc_to_ortho, this->h_from_calc_to_ortho, this->first_cut_sgn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_from_ortho_to_calc, this->h_from_ortho_to_calc, this->nx * this->ny * this->nz * sizeof(int), cudaMemcpyHostToDevice);

    // CUSPARSE & CUBLAS APIs 
    cublasCreate_v2(&this->bandle);
    cusparseCreate(&this->handle);
    cusparseCreateCsr(&this->matA, this->A_num_rows, this->A_num_rows, this->A_nnz, this->dA_csrOffsets, this->dA_columns, this->dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_phi, this->A_num_rows, this->d_phi, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_phi_ortho, this->nx * this->ny * this->nz, this->d_phi_ortho, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_r, this->A_num_rows, this->d_r, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_p, this->A_num_rows, this->d_p, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_Ap, this->A_num_rows, this->d_Ap, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_charge, this->A_num_rows, this->d_charge, CUDA_R_32F);
    cusparseCreateDnVec(&this->vec_charge_ortho, this->nx * this->ny * this->nz,this-> d_charge_ortho, CUDA_R_32F);
    cusparseSpMV_bufferSize(this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &this->minus, this->matA, this->vec_phi, &this->zero, this->vec_r, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &this->bufferSize);
    cudaMalloc(&this->dBuffer, this->bufferSize);
}

void FGM_SURF::Clear_MESH() {
    if (is_initialized == 1) {
		free(hA_csrOffsets);
		free(hA_columns);
		free(hA_values);
		free(h_phi);
		free(h_const);
		free(h_from_calc_to_ortho);
		free(h_from_ortho_to_calc);
		free(h_phi_ortho);
		free(result_phi);
		cudaFree(dA_csrOffsets);
		cudaFree(dA_columns);
		cudaFree(dA_values);
		cudaFree(d_phi);
		cudaFree(d_const);
		cudaFree(d_phi_ortho);
		cudaFree(d_charge);
		cudaFree(d_charge_ortho);
		cudaFree(d_r);
		cudaFree(d_p);
		cudaFree(d_Ap);
		cudaFree(d_from_calc_to_ortho);
		cudaFree(d_from_ortho_to_calc);
		cublasDestroy_v2(bandle);
		cusparseDestroy(handle);
		cusparseDestroySpMat(matA);
		cusparseDestroyDnVec(vec_phi);
		cusparseDestroyDnVec(vec_phi_ortho);
		cusparseDestroyDnVec(vec_r);
		cusparseDestroyDnVec(vec_p);
		cusparseDestroyDnVec(vec_Ap);
		cusparseDestroyDnVec(vec_charge);
		cusparseDestroyDnVec(vec_charge_ortho);
		free(dBuffer);
		free(dBuffer2);
	}
}

void FGM_SURF::Memory_Allocate_SPHERE() 
{
    if (!Malloc_Safely((void**)&(this->h_points), sizeof(float) * this->N_points * 3 ))
        printf("        Error occurs when malloc FGM_SURF::h_points in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Malloc_Safely((void**)&(this->h_edges), sizeof(int) * this->N_edges * 2))
        printf("        Error occurs when malloc FGM_SURF::N_edges in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Malloc_Safely((void**)&(this->h_faces), sizeof(int) * this->N_faces * 3))
        printf("        Error occurs when malloc FGM_SURF::N_edges in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Malloc_Safely((void**)&(this->h_edges_length_unit), sizeof(float) * this->N_edges))
        printf("        Error occurs when malloc FGM_SURF::h_edges_length_unit in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Malloc_Safely((void**)&(this->shape_after_force), sizeof(float) * this->N_points * 3))
        printf("        Error occurs when malloc FGM_SURF::shape_after_force in FGM_SURF::Memory_Allocate_SPHERE");

    if (!Cuda_Malloc_Safely((void**)&this->d_eq_surface_center, sizeof(float) * this->N_eq * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_eq_surface_center in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_points, sizeof(float) * this->N_points * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_points in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_edges, sizeof(int) * this->N_edges * 2))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_edges in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_faces, sizeof(int) * this->N_faces * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_faces in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_edges_length_unit, sizeof(float) * this->N_edges))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_edges_length_unit in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_sphere_phi_each_point, sizeof(float) * this->N_points))
		printf("        Error occurs when CUDA malloc FGM_SURF::d_sphere_phi_each_point in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_sphere_Electric_Field_each_point, sizeof(float) * this->N_points * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_sphere_Electric_Field_each_point in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_E_result, sizeof(float) * 3 * this->n_charge))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_E_result in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_solid_angle, sizeof(float) * this->N_faces))
		printf("        Error occurs when CUDA malloc FGM_SURF::d_solid_angle in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_vec_one, sizeof(float) * this->N_faces))
		printf("        Error occurs when CUDA malloc FGM_SURF::d_vec_one in FGM_SURF::Memory_Allocate_SPHERE");

    if (!Cuda_Malloc_Safely((void**)&this->d_Vel, sizeof(float) * this->N_points * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_Vel in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_Acc, sizeof(float) * this->N_points * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_Acc in FGM_SURF::Memory_Allocate_SPHERE");
    if (!Cuda_Malloc_Safely((void**)&this->d_eq_surface_center, sizeof(float) * this->N_eq * 3))
        printf("        Error occurs when CUDA malloc FGM_SURF::d_eq_surface_center in FGM_SURF::Memory_Allocate_SPHERE");
}

void FGM_SURF::Parameter_Host_To_Device_SPHERE() 
{
    cudaMemcpy(this->d_points, this->h_points, this->N_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_edges, this->h_edges, this->N_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_faces, this->h_faces, this->N_faces * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_edges_length_unit, this->h_edges_length_unit, this->N_edges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_eq_surface_center, this->h_eq_surface_center, this->N_eq * 3 * sizeof(float), cudaMemcpyHostToDevice);
    add_by_constant << <(this->N_faces + 255) / 256, 256 >> > (this->d_vec_one, this->N_faces, 1.0);
    
}


void FGM_SURF::Clear_SPHERE()
{
    if (is_initialized == 1) {
		free(h_points);
		free(h_edges);
		free(h_faces);
		free(h_edges_length_unit);
		free(shape_after_force);
		cudaFree(d_eq_surface_center);
		cudaFree(d_points);
		cudaFree(d_edges);
		cudaFree(d_faces);
		cudaFree(d_edges_length_unit);
		cudaFree(d_sphere_phi_each_point);
		cudaFree(d_sphere_Electric_Field_each_point);
		cudaFree(d_E_result);
		cudaFree(d_solid_angle);
		cudaFree(d_vec_one);
		is_initialized = 0;
	}
}

void FGM_SURF::Potential_file_saver(std::string file_name) {
    float* result_temp; Malloc_Safely((void**)&(result_temp), sizeof(float) * this->A_num_rows);
    cudaMemcpy(result_temp, this->d_phi, this->A_num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    // clean result.txt
    std::ofstream f_clean(file_name);
    // write result to txt
    std::ofstream f_result(file_name);
    for (int i = 0; i < A_num_rows; i++) { f_result << result_temp[i] << std::endl; }
}

void FGM_SURF::Sphere_saver(std::string file_name) {
    float* shpere_temp; Malloc_Safely((void**)&(shpere_temp), sizeof(float) * this->N_points * 3);
    cudaMemcpy(shpere_temp, this->d_points, this->N_points * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // clean result.txt
    std::ofstream f_clean(file_name);
    // write result to txt
    std::ofstream f_result(file_name);
    for (int i = 0; i < N_points; i++) { f_result << shpere_temp[3 * i] << "\t" << shpere_temp[3 * i + 1] << "\t" << shpere_temp[3 * i + 2] << std::endl; }
}



__global__ void refresh_to_zero(float* d_phi_ortho, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_phi_ortho[i] = 0.0;
    }
}

__global__ void multiply_by_constant(float* d_array, int n, float cst) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_array[i] = d_array[i] * cst;
    }
}

__global__ void add_by_constant(float* d_array, int n, float cst) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_array[i] += cst;
    }
}

__global__ void add_by_vector(float* d_array, int N_points, float* d_charge_center, int chg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        d_array[i * 3] += d_charge_center[3 * chg];
        d_array[i * 3 + 1] += d_charge_center[3 * chg + 1];
        d_array[i * 3 + 2] += d_charge_center[3 * chg + 2];
    }
}

__global__ void add_by_vector(float* d_array, int N_points, VECTOR* d_charge_center, int chg) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        d_array[i * 3] += d_charge_center[chg].x;
        d_array[i * 3 + 1] += d_charge_center[chg].y;
        d_array[i * 3 + 2] += d_charge_center[chg].z;
    }
}

__global__ void trans_from_ortho_to_calc(float* d_phi_ortho, float* d_phi, int* d_from_ortho_to_calc, int nx, int ny, int nz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nx * ny * nz) {
        if (d_from_ortho_to_calc[i] != -1) {
            d_phi[d_from_ortho_to_calc[i]] = d_phi_ortho[i];
        }
    }
}

__global__ void trans_from_calc_to_ortho(float* d_phi, float* d_phi_ortho, int* d_from_calc_to_ortho, int first_cut_sgn) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < first_cut_sgn) {
        d_phi_ortho[d_from_calc_to_ortho[i]] = d_phi[i];
    }
}


__global__ void plus_two_vectors(float* array_1, float* array_2, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        array_1[i] += array_2[i];
    }
}


void Charge_Interpolation(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge)
{
    refresh_to_zero << <(fgm_surf.nx * fgm_surf.ny * fgm_surf.nz + 255) / 256, 256 >> > (fgm_surf.d_charge_ortho, fgm_surf.nx * fgm_surf.ny * fgm_surf.nz);
    refresh_to_zero << <(fgm_surf.nx * fgm_surf.ny * fgm_surf.nz + 255) / 256, 256 >> > (fgm_surf.d_charge, fgm_surf.A_num_rows);
    charge_interpolation_cuda << <(fgm_surf.n_charge + 255) / 256, 256 >> > (crd, d_charge, fgm_surf.N_start, fgm_surf.nx, fgm_surf.ny, fgm_surf.nz, fgm_surf.box_x, fgm_surf.box_y, fgm_surf.box_z, fgm_surf.dx, fgm_surf.dy, fgm_surf.dz, fgm_surf.n_charge, fgm_surf.d_charge_ortho);
    trans_from_ortho_to_calc << <(fgm_surf.nx * fgm_surf.ny * fgm_surf.nz + 255) / 256, 256 >> > (fgm_surf.d_charge_ortho, fgm_surf.d_charge, fgm_surf.d_from_ortho_to_calc, fgm_surf.nx, fgm_surf.ny, fgm_surf.nz);
    cublasSaxpy(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.one, fgm_surf.d_const, 1, fgm_surf.d_charge, 1);
}

__global__ void charge_interpolation_cuda(VECTOR* crd, float* charge_q, int N_start, int nx, int ny, int nz, float box_x, float box_y, float box_z,
    float dx, float dy, float dz, int n_charge, float* d_charge_ortho) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_x, idx_y, idx_z; // 电荷所在的正交网格坐标
    int idx_x_plus, idx_y_plus, idx_z_plus; // 电荷所在的正交网格坐标的下一个,符合3d周期性边界条件
    float xx, yy, zz; // 电荷在该正交网格中的归一化位置
    if (i < n_charge) {
        //DEBUG FOR 3DPBC 7-22
        idx_x = floor(crd[i + N_start].x / dx);
        idx_y = floor(crd[i + N_start].y / dy);
        idx_z = floor(crd[i + N_start].z / dz);

        xx = (crd[i + N_start].x - idx_x * dx) / dx;
        yy = (crd[i + N_start].y - idx_y * dy) / dy;
        zz = (crd[i + N_start].z - idx_z * dz) / dz;

        if (idx_x < 0) { idx_x += nx; } idx_x %= nx;
        if (idx_y < 0) { idx_y += ny; } idx_y %= ny;
        if (idx_z < 0) { idx_z += nz; } idx_z %= nz;

        idx_x_plus = (idx_x + 1) % nx;
        idx_y_plus = (idx_y + 1) % ny;
        idx_z_plus = (idx_z + 1) % nz;
        float q = charge_q[i + N_start];

        // 伪逆向三线性插值
        atomicAdd(&d_charge_ortho[idx_x + idx_y * nx + idx_z * nx * ny], (1 - xx) * (1 - yy) * (1 - zz) * q);
        atomicAdd(&d_charge_ortho[idx_x_plus + idx_y * nx + idx_z * nx * ny], xx * (1 - yy) * (1 - zz) * q);
        atomicAdd(&d_charge_ortho[idx_x + idx_y_plus * nx + idx_z * nx * ny], (1 - xx) * yy * (1 - zz) * q);
        atomicAdd(&d_charge_ortho[idx_x_plus + idx_y_plus * nx + idx_z * nx * ny], xx * yy * (1 - zz) * q);
        atomicAdd(&d_charge_ortho[idx_x + idx_y * nx + idx_z_plus * nx * ny], (1 - xx) * (1 - yy) * zz * q);
        atomicAdd(&d_charge_ortho[idx_x_plus + idx_y * nx + idx_z_plus * nx * ny], xx * (1 - yy) * zz * q);
        atomicAdd(&d_charge_ortho[idx_x + idx_y_plus * nx + idx_z_plus * nx * ny], (1 - xx) * yy * zz * q);
        atomicAdd(&d_charge_ortho[idx_x_plus + idx_y_plus * nx + idx_z_plus * nx * ny], xx * yy * zz * q);
    }
}

void CG_Solver(FGM_SURF fgm_surf) {
    // Initialize r = b - Ax
    cusparseSpMV(fgm_surf.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &fgm_surf.minus, fgm_surf.matA, fgm_surf.vec_phi, &fgm_surf.zero, fgm_surf.vec_r, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, fgm_surf.dBuffer);
    cublasSaxpy(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.one, fgm_surf.d_charge, 1, fgm_surf.d_r, 1);
    cudaMemcpy(fgm_surf.d_p, fgm_surf.d_r, fgm_surf.A_num_rows * sizeof(float), cudaMemcpyDeviceToDevice);
    // Initialize err
    cublasSnrm2(fgm_surf.bandle, fgm_surf.A_num_rows, fgm_surf.d_r, 1, &fgm_surf.err);
    // CG-iteration
    int cg_count = 0;
    while (fgm_surf.err >= 1e-5) {
        // alpha = (r, r) / (Ap, p)
        cublasSdot(fgm_surf.bandle, fgm_surf.A_num_rows, fgm_surf.d_r, 1, fgm_surf.d_r, 1, &fgm_surf.r_r);
        cusparseSpMV(fgm_surf.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &fgm_surf.one, fgm_surf.matA, fgm_surf.vec_p, &fgm_surf.zero, fgm_surf.vec_Ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, fgm_surf.dBuffer);
        cublasSdot(fgm_surf.bandle, fgm_surf.A_num_rows, fgm_surf.d_Ap, 1, fgm_surf.d_p, 1, &fgm_surf.p_A_p);
        fgm_surf.alpha = fgm_surf.r_r / fgm_surf.p_A_p;

        // phi = phi + alpha * p;  r = r - alpha * Ap
        cublasSaxpy(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.alpha, fgm_surf.d_p, 1, fgm_surf.d_phi, 1);
        fgm_surf.alpha = -fgm_surf.alpha;
        cublasSaxpy(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.alpha, fgm_surf.d_Ap, 1, fgm_surf.d_r, 1);

        // calc new (r,r) ;  beta = (r,r)_new / (r,r)
        cublasSdot(fgm_surf.bandle, fgm_surf.A_num_rows, fgm_surf.d_r, 1, fgm_surf.d_r, 1, &fgm_surf.r_r_next);
        fgm_surf.beta = fgm_surf.r_r_next / fgm_surf.r_r;

        // calc new p = r + beta * p
        cublasSscal(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.beta, fgm_surf.d_p, 1);
        cublasSaxpy(fgm_surf.bandle, fgm_surf.A_num_rows, &fgm_surf.one, fgm_surf.d_r, 1, fgm_surf.d_p, 1);

        // calc err
        cublasSnrm2(fgm_surf.bandle, fgm_surf.A_num_rows, fgm_surf.d_r, 1, &fgm_surf.err);
        cg_count++;
    }
    printf("CG Iter Loop = %d\n", cg_count);
}


__global__ void boundary_force_shpere(int N_points, float* points, int N_eq, float* boundary_atom_center,
    const float k1, const float VDW_boundary, const float Attach_force, float* Acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        for (int i_eq = 0; i_eq < N_eq; i_eq++)
        {
           // printf("%f\n", boundary_atom_center[0]); //DEBUG
            float dx = points[i * 3] - boundary_atom_center[i_eq * 3];
            float dy = points[i * 3 + 1] - boundary_atom_center[i_eq * 3 + 1];
            float dz = points[i * 3 + 2] - boundary_atom_center[i_eq * 3 + 2];
            float r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r > VDW_boundary) {
                Acc[i * 3] += k1 * dx / pow(r, 3);
                Acc[i * 3 + 1] += k1 * dy / pow(r, 3);
                Acc[i * 3 + 2] += k1 * dz / pow(r, 3);
            }
            else {
                Acc[i * 3] += k1 * dx * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
                Acc[i * 3 + 1] += k1 * dy * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
                Acc[i * 3 + 2] += k1 * dz * (1 / pow(r, 2) + Attach_force * pow(VDW_boundary - r, 1) / r);
            }
        }
    }
}


__global__ void triangle_force(float* points, int* faces, const float k2, float* Acc, int N_faces) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_faces) {
        float x1 = points[faces[i * 3] * 3];
        float y1 = points[faces[i * 3] * 3 + 1];
        float z1 = points[faces[i * 3] * 3 + 2];
        float x2 = points[faces[i * 3 + 1] * 3];
        float y2 = points[faces[i * 3 + 1] * 3 + 1];
        float z2 = points[faces[i * 3 + 1] * 3 + 2];
        float x3 = points[faces[i * 3 + 2] * 3];
        float y3 = points[faces[i * 3 + 2] * 3 + 1];
        float z3 = points[faces[i * 3 + 2] * 3 + 2];
        float nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
        float ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
        float nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        float r = sqrt(nx * nx + ny * ny + nz * nz);
        nx /= r; ny /= r; nz /= r;
        Acc[faces[i * 3] * 3] += k2 * nx;
        Acc[faces[i * 3] * 3 + 1] += k2 * ny;
        Acc[faces[i * 3] * 3 + 2] += k2 * nz;
        Acc[faces[i * 3 + 1] * 3] += k2 * nx;
        Acc[faces[i * 3 + 1] * 3 + 1] += k2 * ny;
        Acc[faces[i * 3 + 1] * 3 + 2] += k2 * nz;
        Acc[faces[i * 3 + 2] * 3] += k2 * nx;
        Acc[faces[i * 3 + 2] * 3 + 1] += k2 * ny;
        Acc[faces[i * 3 + 2] * 3 + 2] += k2 * nz;
    }
}

__global__ void edge_force(float* points, int* edges, float* edge_length_init,
    const float k3, const float k4, float* Acc, int N_edges, float R) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_edges) {
        float x1 = points[edges[i * 2] * 3];
        float y1 = points[edges[i * 2] * 3 + 1];
        float z1 = points[edges[i * 2] * 3 + 2];
        float x2 = points[edges[i * 2 + 1] * 3];
        float y2 = points[edges[i * 2 + 1] * 3 + 1];
        float z2 = points[edges[i * 2 + 1] * 3 + 2];
        float dx = x1 - x2;
        float dy = y1 - y2;
        float dz = z1 - z2;
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        float F = k4 * (edge_length_init[i] * R - r);
        if (edge_length_init[i] * R > r) { F += k3 * (edge_length_init[i] * R - r); }
        Acc[edges[i * 2] * 3] += F * dx;
        Acc[edges[i * 2] * 3 + 1] += F * dy;
        Acc[edges[i * 2] * 3 + 2] += F * dz;
        Acc[edges[i * 2 + 1] * 3] -= F * dx;
        Acc[edges[i * 2 + 1] * 3 + 1] -= F * dy;
        Acc[edges[i * 2 + 1] * 3 + 2] -= F * dz;
    }
}

__global__ void center_force(int N_points, float* points, VECTOR* crd,
    const float k5, const float R, float* Acc, int chg) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        float dx = points[i * 3] - crd[chg].x;
        float dy = points[i * 3 + 1] - crd[chg].y;
        float dz = points[i * 3 + 2] - crd[chg].z;
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        float F = -k5 / r;
        if (R > r) { F = k5 / r; }
        Acc[i * 3] += F * dx;
        Acc[i * 3 + 1] += F * dy;
        Acc[i * 3 + 2] += F * dz;
    }
}

__global__ void update_position_Leapfrog(int N_points, float* points, float* Vel, const float dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        points[i * 3] += Vel[i * 3] * dt;
        points[i * 3 + 1] += Vel[i * 3 + 1] * dt;
        points[i * 3 + 2] += Vel[i * 3 + 2] * dt;
    }
}

__global__ void update_velocity_Leapfrog(int N_points, float* Vel, float* Acc, const float dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_points) {
        Vel[i * 3] += Acc[i * 3] * dt;
        Vel[i * 3 + 1] += Acc[i * 3 + 1] * dt;
        Vel[i * 3 + 2] += Acc[i * 3 + 2] * dt;
    }
}


void Sphere_Autoconfort_Iterator(FGM_SURF fgm_surf, VECTOR* crd, int chg) {
    // 初始化采样球面位置
    cudaMemcpy(fgm_surf.d_points, fgm_surf.h_points, fgm_surf.N_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    multiply_by_constant << <(fgm_surf.N_points * 3 + 255) / 256, 256 >> > (fgm_surf.d_points, fgm_surf.N_points * 3, fgm_surf.R);
    add_by_vector << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.d_points, fgm_surf.N_points, crd, chg);
    // 初始化速度
    refresh_to_zero << <(fgm_surf.N_points * 3 + 255) / 256, 256 >> > (fgm_surf.d_Vel, fgm_surf.N_points * 3);

    // 动力学演化自适应曲面
    for (int i = 0; i < fgm_surf.max_iter; i++) {
        // 重置加速度
        refresh_to_zero << <(fgm_surf.N_points * 3 + 255) / 256, 256 >> > (fgm_surf.d_Acc, fgm_surf.N_points * 3);
        // 计算力
        boundary_force_shpere << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.N_points, fgm_surf.d_points, fgm_surf.N_eq, fgm_surf.d_eq_surface_center, fgm_surf.k1, fgm_surf.VDW_boundary, fgm_surf.Attach_force, fgm_surf.d_Acc);
        triangle_force << <(fgm_surf.N_faces + 255) / 256, 256 >> > (fgm_surf.d_points, fgm_surf.d_faces, fgm_surf.k2, fgm_surf.d_Acc, fgm_surf.N_faces);
        edge_force << <(fgm_surf.N_edges + 255) / 256, 256 >> > (fgm_surf.d_points, fgm_surf.d_edges, fgm_surf.d_edges_length_unit, fgm_surf.k3, fgm_surf.k4, fgm_surf.d_Acc, fgm_surf.N_edges, fgm_surf.R);
        center_force << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.N_points, fgm_surf.d_points, crd, fgm_surf.k5, fgm_surf.R, fgm_surf.d_Acc, chg);

        // 更新速度
        update_velocity_Leapfrog << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.N_points, fgm_surf.d_Vel, fgm_surf.d_Acc, fgm_surf.dt_shpere);
        // 引入摩擦
        multiply_by_constant << <(fgm_surf.N_points * 3 + 255) / 256, 256 >> > (fgm_surf.d_Vel, fgm_surf.N_points * 3, fgm_surf.gamma);
        // 更新位置
        update_position_Leapfrog << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.N_points, fgm_surf.d_points, fgm_surf.d_Vel, fgm_surf.dt_shpere);
    }
}




// 重载::面向对象
void Eletric_Field_Calculator(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge, int chg, const int* excluded_list_start, const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc)
{
    Potential_and_Elecfield_calculator << <(fgm_surf.N_points + 255) / 256, 256 >> > (fgm_surf.d_phi, fgm_surf.d_from_ortho_to_calc, fgm_surf.N_points, fgm_surf.d_points, fgm_surf.nx, fgm_surf.ny, fgm_surf.nz, fgm_surf.box_x, fgm_surf.box_y, fgm_surf.box_z, fgm_surf.d_sphere_phi_each_point, fgm_surf.d_sphere_Electric_Field_each_point);
    Calc_Green_Integral << <(fgm_surf.N_faces + 255) / 256, 256 >> > (fgm_surf.N_faces, fgm_surf.d_faces, fgm_surf.d_points, chg, crd, fgm_surf.d_sphere_phi_each_point, fgm_surf.d_sphere_Electric_Field_each_point, fgm_surf.d_E_result, fgm_surf.N_start);
    for (int chg_other = 0; chg_other < fgm_surf.n_charge; chg_other++) {   // 遍历所有电荷
        bool is_exclude = 0; 
        for (int i=0; i< excluded_atom_numbers[chg + fgm_surf.N_start]; i++)   // 检测exclude
        {
            if (chg_other == excluded_list[excluded_list_start[chg + fgm_surf.N_start] + i] - fgm_surf.N_start) { is_exclude = 1; }
        }
        if (chg_other != chg && is_exclude == 0) {
            float tot_angle = 0;
            Calculate_Solid_Angle_in_current_Hull << <(fgm_surf.N_faces + 255) / 256, 256 >> > (fgm_surf.N_faces, fgm_surf.d_faces, fgm_surf.d_points, chg, chg_other, crd, fgm_surf.d_solid_angle, fgm_surf.box_x, fgm_surf.box_y, fgm_surf.box_z, fgm_surf.N_start);
            cublasSdot(fgm_surf.bandle, fgm_surf.N_faces, fgm_surf.d_solid_angle, 1, fgm_surf.d_vec_one, 1, &tot_angle);
            // DEBUG 7-17
            if (tot_angle > 3.99) { // inside the hull
                Calc_charge_inside << <1, 1 >> > (chg, chg_other, crd, d_charge, fgm_surf.d_E_result, fgm_surf.box_x, fgm_surf.box_y, fgm_surf.box_z, fgm_surf.N_start);
            }
        }
    }
    Calc_FGM_Force_by_Elec_field<< <(fgm_surf.n_charge + 255)/256, 256>> >(fgm_surf.n_charge, fgm_surf.N_start, d_charge, frc, fgm_surf.d_E_result);
}


__global__ void Potential_and_Elecfield_calculator(float* phi, int* from_ortho_to_calc,
    int N_points, float* sphere_crd,
    int nx, int ny, int nz,
    float box_x, float box_y, float box_z,
    float* sphere_phi_each_point, float* sphere_Electric_Field_each_point)
{
    int ker = blockIdx.x * blockDim.x + threadIdx.x;
    if (ker < N_points) {
        float box_dx = box_x / nx; float box_dy = box_y / ny; float box_dz = box_z / nz;
        float x = sphere_crd[3 * ker]; float y = sphere_crd[3 * ker + 1]; float z = sphere_crd[3 * ker + 2];
        int i = floor(x / box_dx); int j = floor(y / box_dy); int k = floor(z / box_dz);
        float dx = x / box_dx - i;
        float dy = y / box_dy - j;
        float dz = z / box_dz - k;
        if (i < 0) { i += nx; }  i %= nx;
        if (j < 0) { j += ny; }  j %= ny;
        if (k < 0) { k += nz; }  k %= nz;
        //printf("xyz = (%f, %f, %f)\t dxdydz = (%f, %f, %f)\t transform to  %d %d %d \n", x, y, z, dx, dy, dz, i, j, k);
        float Phi1 = phi[from_ortho_to_calc[i + nx * j + nx * ny * k]];
        float Phi2 = phi[from_ortho_to_calc[(i + 1) + nx * j + nx * ny * k]];
        float Phi3 = phi[from_ortho_to_calc[(i + 1) + nx * (j + 1) + nx * ny * k]];
        float Phi4 = phi[from_ortho_to_calc[i + nx * (j + 1) + nx * ny * k]];
        float Phi5 = phi[from_ortho_to_calc[i + nx * j + nx * ny * (k + 1)]];
        float Phi6 = phi[from_ortho_to_calc[(i + 1) + nx * j + nx * ny * (k + 1)]];
        float Phi7 = phi[from_ortho_to_calc[(i + 1) + nx * (j + 1) + nx * ny * (k + 1)]];
        float Phi8 = phi[from_ortho_to_calc[i + nx * (j + 1) + nx * ny * (k + 1)]];
        sphere_phi_each_point[ker] = (1 - dx) * (1 - dy) * (1 - dz) * Phi1 + dx * (1 - dy) * (1 - dz) * Phi2 + dx * dy * (1 - dz) * Phi3 + (1 - dx) * dy * (1 - dz) * Phi4 +
            (1 - dx) * (1 - dy) * dz * Phi5 + dx * (1 - dy) * dz * Phi6 + dx * dy * dz * Phi7 + (1 - dx) * dy * dz * Phi8;

        sphere_Electric_Field_each_point[3 * ker] = -Phi1 * (1 - dy) * (1 - dz) + Phi2 * (1 - dy) * (1 - dz) + Phi3 * dy * (1 - dz) - Phi4 * dy * (1 - dz) - Phi5 * (1 - dy) * dz + Phi6 * (1 - dy) * dz + Phi7 * dy * dz - Phi8 * dy * dz;
        sphere_Electric_Field_each_point[3 * ker + 1] = -Phi1 * (1 - dx) * (1 - dz
            ) - Phi2 * dx * (1 - dz) + Phi3 * dx * (1 - dz) + Phi4 * (1 - dx) * (1 - dz) - Phi5 * (1 - dx) * dz - Phi6 * dx * dz + Phi7 * dx * dz + Phi8 * (1 - dx) * dz;
        sphere_Electric_Field_each_point[3 * ker + 2] = -Phi1 * (1 - dx) * (1 - dy) - Phi2 * dx * (1 - dy) - Phi3 * dx * dy - Phi4 * (1 - dx) * dy + Phi5 * (1 - dx) * (1 - dy) + Phi6 * dx * (1 - dy) + Phi7 * dx * dy + Phi8 * (1 - dx) * dy;
        sphere_Electric_Field_each_point[3 * ker] *= -1;
        sphere_Electric_Field_each_point[3 * ker + 1] *= -1;
        sphere_Electric_Field_each_point[3 * ker + 2] *= -1;
    }
}

// Debug: 计算所有立体角并存储在立体角数组 Solid-angle—sum;
__global__ void Calculate_Solid_Angle_in_current_Hull(int N_faces, int* faces, float* sphere_crd,
    int chg, int chg_other, VECTOR* crd, float* d_solid_angle, float box_x, float box_y, float box_z, int N_start)
{
    int ker = blockIdx.x * blockDim.x + threadIdx.x;
    if (ker < N_faces) {
        // 其他电荷坐标
        float px = crd[chg_other + N_start].x; float py = crd[chg_other + N_start].y; float pz = crd[chg_other + N_start].z;
        // 球面积分电荷坐标
        float rx = crd[chg + N_start].x; float ry = crd[chg + N_start].y; float rz = crd[chg + N_start].z;
        // Find Nearest px refer to rx in 3DPBC Conditions DEBUG 7-17   
        px -= int((px - rx) / (0.5 * box_x)) * box_x;
        py -= int((py - ry) / (0.5 * box_y)) * box_y;
        pz -= int((pz - rz) / (0.5 * box_z)) * box_z;

        int i = faces[3 * ker]; int j = faces[3 * ker + 1]; int k = faces[3 * ker + 2];
        float x1 = sphere_crd[3 * i] - px; float y1 = sphere_crd[3 * i + 1] - py; float z1 = sphere_crd[3 * i + 2] - pz;
        float x2 = sphere_crd[3 * j] - px; float y2 = sphere_crd[3 * j + 1] - py; float z2 = sphere_crd[3 * j + 2] - pz;
        float x3 = sphere_crd[3 * k] - px; float y3 = sphere_crd[3 * k + 1] - py; float z3 = sphere_crd[3 * k + 2] - pz;
        float l_1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
        float l_2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2);
        float l_3 = sqrt(x3 * x3 + y3 * y3 + z3 * z3);
        float across_x = y1 * z2 - z1 * y2; float across_y = z1 * x2 - x1 * z2; float across_z = x1 * y2 - y1 * x2;
        float dot_up = x3 * across_x + y3 * across_y + z3 * across_z;
        float dot_down = (l_1 * l_2 * l_3) + (x1 * x2 + y1 * y2 + z1 * z2) * l_3 + (x1 * x3 + y1 * y3 + z1 * z3) * l_2 + (x2 * x3 + y2 * y3 + z2 * z3) * l_1;
        d_solid_angle[ker] = 2 * atan2(dot_up, dot_down) / CONSTANT_Pi;
    }
}

// Need Debug 7-17
__global__ void Calc_charge_inside(int chg, int chg_other, VECTOR* crd, float* d_charge, float* E_result, float box_x, float box_y, float box_z, int N_start) {
    int ker = blockIdx.x * blockDim.x + threadIdx.x;
    if (ker == 0) {
        float rx = crd[chg + N_start].x - crd[chg_other + N_start].x;         rx -= int(rx / (0.5 * box_x)) * box_x;
        float ry = crd[chg + N_start].y - crd[chg_other + N_start].y;         ry -= int(ry / (0.5 * box_y)) * box_y;
        float rz = crd[chg + N_start].z - crd[chg_other + N_start].z;         rz -= int(rz / (0.5 * box_z)) * box_z;
        float r = sqrt(rx * rx + ry * ry + rz * rz);
        // DEBUG 7-10
        E_result[3 * chg] += d_charge[chg_other + N_start] * rx / pow(r, 3);
        E_result[3 * chg + 1] += d_charge[chg_other + N_start] * ry / pow(r, 3);
        E_result[3 * chg + 2] += d_charge[chg_other + N_start] * rz / pow(r, 3);
    }
}

__global__ void Calc_Green_Integral(int N_faces, int* faces, float* sphere_crd, int chg, VECTOR* crd,
    float* sphere_phi_each_point, float* sphere_Electric_Field_each_point, float* E_result, int N_start)
{
    int ker = blockIdx.x * blockDim.x + threadIdx.x;
    if (ker < N_faces) {
        int i = faces[3 * ker]; int j = faces[3 * ker + 1]; int k = faces[3 * ker + 2];
        float x1 = sphere_crd[3 * i]; float y1 = sphere_crd[3 * i + 1]; float z1 = sphere_crd[3 * i + 2];
        float x2 = sphere_crd[3 * j]; float y2 = sphere_crd[3 * j + 1]; float z2 = sphere_crd[3 * j + 2];
        float x3 = sphere_crd[3 * k]; float y3 = sphere_crd[3 * k + 1]; float z3 = sphere_crd[3 * k + 2];

        float Ex = (sphere_Electric_Field_each_point[3 * i] + sphere_Electric_Field_each_point[3 * j] + sphere_Electric_Field_each_point[3 * k]) / 3;
        float Ey = (sphere_Electric_Field_each_point[3 * i + 1] + sphere_Electric_Field_each_point[3 * j + 1] + sphere_Electric_Field_each_point[3 * k + 1]) / 3;
        float Ez = (sphere_Electric_Field_each_point[3 * i + 2] + sphere_Electric_Field_each_point[3 * j + 2] + sphere_Electric_Field_each_point[3 * k + 2]) / 3;
        float phi = (sphere_phi_each_point[i] + sphere_phi_each_point[j] + sphere_phi_each_point[k]) / 3;

        float nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
        float ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
        float nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        float S = 0.5 * sqrt(nx * nx + ny * ny + nz * nz);
        nx /= 2 * S; ny /= 2 * S; nz /= 2 * S;

        float vec_x = (x1 + x2 + x3) / 3 - crd[chg + N_start].x; float vec_y = (y1 + y2 + y3) / 3 - crd[chg + N_start].y; float vec_z = (z1 + z2 + z3) / 3 - crd[chg + N_start].z;
        float r = sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

        atomicAdd(&E_result[chg * 3], 1 / pow(r, 3) * phi * S * nx);
        atomicAdd(&E_result[chg * 3 + 1], 1 / pow(r, 3) * phi * S * ny);
        atomicAdd(&E_result[chg * 3 + 2], 1 / pow(r, 3) * phi * S * nz);

        float Add_const = 1 / pow(r, 3) * S * (Ex * nx + Ey * ny + Ez * nz - 3 * phi / pow(r, 2) * (vec_x * nx + vec_y * ny + vec_z * nz));
        atomicAdd(&E_result[chg * 3], Add_const * vec_x);
        atomicAdd(&E_result[chg * 3 + 1], Add_const * vec_y);
        atomicAdd(&E_result[chg * 3 + 2], Add_const * vec_z);
    }
}

__global__ void Calc_FGM_Force_by_Elec_field(const int n_charge, const int N_start, float* d_charge, VECTOR* frc, float* E_result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_charge) {
        frc[i + N_start].x -= d_charge[i + N_start] * E_result[3 * i];
        frc[i + N_start].y -= d_charge[i + N_start] * E_result[3 * i + 1];
        frc[i + N_start].z -= d_charge[i + N_start] * E_result[3 * i + 2];
    }
}

void FGM_CalC_Force_Only_with_Exclude(FGM_SURF fgm_surf, VECTOR* crd, float* d_charge, 
    const int* excluded_list_start, const int* excluded_list, const int* excluded_numbers, VECTOR* frc)
{
    Charge_Interpolation(fgm_surf, crd, d_charge);
    CG_Solver(fgm_surf);
    refresh_to_zero << <(fgm_surf.n_charge * 3 + 255), 256 >> > (fgm_surf.d_E_result, fgm_surf.n_charge * 3);  //重置电场
    for (int i = 0; i < fgm_surf.n_charge; i++)
    {
        Sphere_Autoconfort_Iterator(fgm_surf, crd, fgm_surf.N_start + 3);
        Eletric_Field_Calculator(fgm_surf, crd, d_charge, 3, excluded_list_start, excluded_list, excluded_numbers, frc);
    }
}
