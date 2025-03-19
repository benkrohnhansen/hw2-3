#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

#define bin_size cutoff
#define cutoff_squared (cutoff * cutoff)
#define min_r_squared (min_r * min_r)

int blks;
int num_bins;

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* sorted_parts, int* bin_offsets, int num_bins) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_parts) return;

	int particle_idx = sorted_parts[tid];
	particle_t& p = particles[particle_idx];

	double ax = 0;
	double ay = 0;

	double px = p.x;
	double py = p.y;

	int x = px / bin_size;
	int y = py / bin_size;

	int start_x = max(0, x - 1);
	int end_x = min(num_bins - 1, x + 1);
	int start_y = max(0, y - 1);
	int end_y = min(num_bins - 1, y + 1);

	for (int i = start_x; i <= end_x; i += 1) {
    	for (int j = start_y; j <= end_y; j += 1) {
        	int neighbor_bin = i * num_bins + j;
        	for (int k = neighbor_bin == 0 ? 0 : bin_offsets[neighbor_bin - 1]; k < bin_offsets[neighbor_bin]; k += 1) {
            	int neighbor_idx = sorted_parts[k];
            	if (particle_idx != neighbor_idx) {
                	particle_t& n = particles[neighbor_idx];

                	double dx = n.x - px;
                	double dy = n.y - py;
                	double r2 = dx * dx + dy * dy;
                	if (r2 > cutoff_squared) continue;
                	r2 = (r2 > min_r_squared) ? r2 : min_r_squared;
                	double r = sqrt(r2);
                	double coef = (1 - cutoff / r) / r2 / mass;
                	ax += coef * dx;
                	ay += coef * dy;
            	}
        	}
    	}
	}

	p.ax = ax;
	p.ay = ay;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_parts) return;
	particle_t* p = &particles[tid];
	p->vx += p->ax * dt;
	p->vy += p->ay * dt;
	p->x += p->vx * dt;
	p->y += p->vy * dt;
	while (p->x < 0 || p->x > size) {
    	p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
    	p->vx = -(p->vx);
	}
	while (p->y < 0 || p->y > size) {
    	p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
    	p->vy = -(p->vy);
	}
}

__global__ void compute_parts_per_bin(particle_t* parts, int num_parts, int* parts_per_bin, int num_bins) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_parts) return;
	int x = parts[tid].x / bin_size;
	int y = parts[tid].y / bin_size;
	atomicAdd(&parts_per_bin[x * num_bins + y], 1);
}

__global__ void assign_parts_to_bins(particle_t* parts, int num_parts, int* sorted_parts, int* bin_positions, int num_bins) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_parts) return;
	int x = parts[tid].x / bin_size;
	int y = parts[tid].y / bin_size;
	int pos = atomicAdd(&bin_positions[x * num_bins + y], 1);
	sorted_parts[pos] = tid;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	// parts live in GPU memory
	blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
	num_bins = static_cast<int>(ceil(size / bin_size));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

	// Making our device_vectors static helps us to avoid repeated cudaMalloc calls

	static thrust::device_vector<int> parts_per_bin(num_bins * num_bins);
	thrust::fill(parts_per_bin.begin(), parts_per_bin.end(), 0);
	compute_parts_per_bin<<<blks, NUM_THREADS>>>(parts, num_parts, thrust::raw_pointer_cast(parts_per_bin.data()), num_bins);

	static thrust::device_vector<int> bin_offsets(num_bins * num_bins);
	thrust::fill(bin_offsets.begin(), bin_offsets.end() - 1, 0);
	thrust::fill(bin_offsets.end() - 1, bin_offsets.end(), num_parts);
	thrust::exclusive_scan(parts_per_bin.begin(), parts_per_bin.end(), bin_offsets.begin());

	static thrust::device_vector<int> sorted_parts(num_parts);
	static thrust::device_vector<int> bin_positions(num_bins * num_bins);
	assign_parts_to_bins<<<blks, NUM_THREADS>>>(parts, num_parts, thrust::raw_pointer_cast(sorted_parts.data()), thrust::raw_pointer_cast(bin_offsets.data()), num_bins);

	compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, thrust::raw_pointer_cast(sorted_parts.data()), thrust::raw_pointer_cast(bin_offsets.data()), num_bins);
	move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
