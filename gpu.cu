#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256
#define CELL_SIZE 1.2 * cutoff  // 셀 크기 설정 (Cutoff 거리보다 약간 더 크게)


// Put any static global variables here that you will use throughout the simulation.
int blks; // cuda에서 사용할 block 수
int num_cells_x, num_cells_y;
int* d_cell_starts;
int* d_cell_ends;
particle_t* d_ghost_particles;
int* d_ghost_counts;

// device: GPU 에서 호출하고 실행할 함수, return 값 상관 x 
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    // 거리 차이를 계산
    double dx = neighbor.x - particle.x; 
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) // cut off보다 멀면 상관 x
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r; // min_r 보다 큰지 작은지 판단해서 큰 애로 r2 에 저장
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass; // repulsive force 계산
    particle.ax += coef * dx; // 가속도는 coef x 거리 차이, 힘 / 질량
    particle.ay += coef * dy;

}

// CPU 에서 호출, GPU 에서 실행, void type 만 가능
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* cell_starts, int* cell_ends, int num_cells_x, int num_cells_y) {
    // Get thread (particle) ID
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;  // thread id를 알아야돼
    
    if (tx >= num_cells_x || ty >= num_cells_y)
        return;

    int cell_index = tx + ty * num_cells_x; // cell index that this thread is responsible for
    
    int start = cell_starts[cell_index]; // 해당 cell의 시작 index
    int end = cell_ends[cell_index]; // 해당 cell의 끝 index

    // particles[tid].ax = particles[tid].ay = 0;

    for (int dx = -1; dx <= 1; dx++){
        for (int dy = -1; dy <= 1; dy++){

            // neighbor cell의 x y index
            int neighbor_x = tx + dx; 
            int neighbor_y = ty + dy;

            if (neighbor_x < 0 || neighbor_x >= num_cells_x || neighbor_y < 0 || neighbor_y >= num_cells_y)
                continue;

            int neighbor_index = neighbor_x + neighbor_y * num_cells_x; // neighbor cell의 index
            int neighbor_start = cell_starts[neighbor_index]; // neighbor cell의 시작 index
            int neighbor_end = cell_ends[neighbor_index]; // neighbor cell의 끝 index
            
            for (int i = start; i < end; i++){
                for (int j = neighbor_start; j < neighbor_end; j++){
                    if (i != j) {
                        apply_force_gpu(particles[i], particles[j]);
                    
                    }
                }   
            }
        }
    }
}

__global__ void compute_forces_with_ghosts(particle_t* parts, int num_parts, particle_t* ghost_particles, int* ghost_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& p = parts[tid];

    for (int i = 0; i < ghost_counts[blockIdx.x]; i++) {
        apply_force_gpu(p, ghost_particles[i]);
    }
}


__global__ void move_gpu(particle_t* particles, int num_parts, double size, particle_t* ghost_particles, int* ghost_counts) {


    // Get thread (particle) ID
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;  

    int tid = tx + ty * gridDim.x * blockDim.x; // thread id를 알아야돼
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt; // p는 vx를 가르키는 포인터(주소값) 라는 뜻이다
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }

    // ghost particle
    if (p->x < 0 || p->x > size || p->y < 0 || p->y > size) {
        // atomicAdd: 여러 스레드가 동시에 접근해도 괜찮게 함
        int idx = atomicAdd(&ghost_counts[blockIdx.x], 1); // 값을 1 증가시킴
        // 도메인 밖으로 나간 입자의 포인터를 저장
        ghost_particles[idx] = *p;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    // 1️⃣ 블록 개수 계산 (NUM_THREADS 개의 스레드 블록으로 나눔)
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

      // 2️⃣ 도메인(Cell Grid) 설정
    num_cells_x = (int)(size / CELL_SIZE);  // X축 셀 개수
    num_cells_y = (int)(size / CELL_SIZE);  // Y축 셀 개수
    int total_cells = num_cells_x * num_cells_y; // 전체 셀 개수
  
    // 3️⃣ CUDA 메모리 할당
    cudaMalloc((void**)&d_cell_starts, total_cells * sizeof(int)); // 각 셀의 첫 번째 입자 인덱스
    cudaMalloc((void**)&d_cell_ends, total_cells * sizeof(int));   // 각 셀의 마지막 입자 인덱스
    cudaMalloc((void**)&d_ghost_particles, num_parts * sizeof(particle_t)); // 고스트 입자 저장 공간
    cudaMalloc((void**)&d_ghost_counts, blks * sizeof(int));  // 각 블록에서 생성된 고스트 입자 개수 저장
  
    // 4️⃣ `cell_starts`, `cell_ends`, `ghost_counts` 초기화
    cudaMemset(d_cell_starts, 0x7FFFFFFF, total_cells * sizeof(int)); // 최소값 설정
    cudaMemset(d_cell_ends, 0, total_cells * sizeof(int)); // 최대값 설정
    cudaMemset(d_ghost_counts, 0, blks * sizeof(int)); // 고스트 입자 개수 초기화
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    dim3 blockDim(16, 16);
    dim3 gridDim((num_cells_x + blockDim.x - 1) / blockDim.x,
                 (num_cells_y + blockDim.y - 1) / blockDim.y);
    // 1️⃣ 입자 간 힘 계산 (각 셀과 주변 8개 셀의 입자들과 상호작용)
    compute_forces_gpu<<<gridDim, blockDim>>>(parts, num_parts, d_cell_starts, d_cell_ends, num_cells_x, num_cells_y);
    cudaDeviceSynchronize();

    // 2️⃣ 입자 이동 (경계를 넘는 입자는 `ghost_particles`에 저장)
    move_gpu<<<gridDim, blockDim>>>(parts, num_parts, size, d_ghost_particles, d_ghost_counts);
    cudaDeviceSynchronize();

    // 3️⃣ 고스트 입자 리스트를 사용하여 추가 힘 계산
    compute_forces_with_ghosts<<<gridDim, blockDim>>>(parts, num_parts, d_ghost_particles, d_ghost_counts);
    cudaDeviceSynchronize();
}