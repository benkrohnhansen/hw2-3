#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256
#define bin_size cutoff
#define cutoff_squared (cutoff * cutoff)
#define min_r_squared (min_r * min_r)

int num_bins_x, num_bins_y;

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* sorted_parts, int* bin_offsets, int num_bins_x, int num_bins_y) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx >= num_bins_x || ty >= num_bins_y) return;

    int bin_index = tx + ty * num_bins_x;
    int start = (bin_index == 0) ? 0 : bin_offsets[bin_index - 1];
    int end = bin_offsets[bin_index];

    for (int i = start; i < end; i++) {
        int particle_idx = sorted_parts[i];
        particle_t& p = particles[particle_idx];

        p.ax = 0;
        p.ay = 0;

        double px = p.x;
        double py = p.y;

        int start_x = max(0, tx - 1);
        int end_x = min(num_bins_x - 1, tx + 1);
        int start_y = max(0, ty - 1);
        int end_y = min(num_bins_y - 1, ty + 1);

        for (int nx = start_x; nx <= end_x; nx++) {
            for (int ny = start_y; ny <= end_y; ny++) {
                int neighbor_bin = nx + ny * num_bins_x;
                int neighbor_start = (neighbor_bin == 0) ? 0 : bin_offsets[neighbor_bin - 1];
                int neighbor_end = bin_offsets[neighbor_bin];

                for (int j = neighbor_start; j < neighbor_end; j++) {
                    int neighbor_idx = sorted_parts[j];
                    if (particle_idx != neighbor_idx) {
                        particle_t& n = particles[neighbor_idx];

                        double dx = n.x - px;
                        double dy = n.y - py;
                        double r2 = dx * dx + dy * dy;
                        if (r2 > cutoff_squared) continue;

                        r2 = max(r2, min_r_squared);
                        double r = sqrt(r2);
                        double coef = (1 - cutoff / r) / r2 / mass;
                        p.ax += coef * dx;
                        p.ay += coef * dy;
                    }
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t* p = &particles[tid];

    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    if (p->x < 0) {
        p->x = -p->x;
        p->vx = -p->vx;
    }
    if (p->x > size) {
        p->x = 2 * size - p->x;
        p->vx = -p->vx;
    }
    if (p->y < 0) {
        p->y = -p->y;
        p->vy = -p->vy;
    }
    if (p->y > size) {
        p->y = 2 * size - p->y;
        p->vy = -p->vy;
    }
}

__global__ void compute_parts_per_bin(particle_t* parts, int num_parts, int* parts_per_bin, int num_bins_x, int num_bins_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int x = parts[tid].x / bin_size;
    int y = parts[tid].y / bin_size;
    atomicAdd(&parts_per_bin[x + y * num_bins_x], 1);
}

__global__ void assign_parts_to_bins(particle_t* parts, int num_parts, int* sorted_parts, int* bin_positions, int num_bins_x, int num_bins_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int x = parts[tid].x / bin_size;
    int y = parts[tid].y / bin_size;
    int pos = atomicAdd(&bin_positions[x + y * num_bins_x], 1);
    sorted_parts[pos] = tid;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    num_bins_x = static_cast<int>(ceil(size / bin_size));
    num_bins_y = static_cast<int>(ceil(size / bin_size));
}

void simulate_one_step(particle_t* parts, int num_parts, double size,
		       double& total_comp_time, double& total_thrust_time) {
    cudaEvent_t start_thrust, stop_thrust, start_compute, stop_compute;
    float time_thrust = 0.0f, time_compute = 0.0f;

    cudaEventCreate(&start_thrust);
    cudaEventCreate(&stop_thrust);
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);

    // ======== START THRUST TIMING =========
    cudaEventRecord(start_thrust, 0);

    static thrust::device_vector<int> parts_per_bin(num_bins_x * num_bins_y);
    thrust::fill(parts_per_bin.begin(), parts_per_bin.end(), 0);
    compute_parts_per_bin<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
        parts, num_parts, thrust::raw_pointer_cast(parts_per_bin.data()), num_bins_x, num_bins_y);

    static thrust::device_vector<int> bin_offsets(num_bins_x * num_bins_y);
    thrust::fill(bin_offsets.begin(), bin_offsets.end() - 1, 0);
    thrust::fill(bin_offsets.end() - 1, bin_offsets.end(), num_parts);
    thrust::exclusive_scan(parts_per_bin.begin(), parts_per_bin.end(), bin_offsets.begin());

    static thrust::device_vector<int> sorted_parts(num_parts);
    static thrust::device_vector<int> bin_positions(num_bins_x * num_bins_y);
 
    //=========== STOP THRUST TIMING ========= 
    cudaEventRecord(stop_thrust, 0); 
    cudaEventSynchronize(stop_thrust);
    cudaEventElapsedTime(&time_thrust, start_thrust, stop_thrust);

    total_thrust_time += time_thrust;

    //========== START COMPUTE TIMING ========
    cudaEventRecord(start_compute, 0);

   assign_parts_to_bins<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
        parts, num_parts, thrust::raw_pointer_cast(sorted_parts.data()), thrust::raw_pointer_cast(bin_offsets.data()), num_bins_x, num_bins_y);

    dim3 blockDim(16, 16);
    dim3 gridDim((num_bins_x + blockDim.x - 1) / blockDim.x,
                 (num_bins_y + blockDim.y - 1) / blockDim.y);

    compute_forces_gpu<<<gridDim, blockDim>>>(
        parts, num_parts, thrust::raw_pointer_cast(sorted_parts.data()), thrust::raw_pointer_cast(bin_offsets.data()), num_bins_x, num_bins_y);

    move_gpu<<<(num_parts + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(parts, num_parts, size);

    //========== STOP COMPUTE TIMING =========
    cudaEventRecord(stop_compute, 0);
    cudaEventSynchronize(stop_compute);
    cudaEventElapsedTime(&time_compute, start_compute, stop_compute);

    total_comp_time += time_compute;

    cudaEventDestroy(start_thrust);
    cudaEventDestroy(stop_thrust);
    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);
}


// 
