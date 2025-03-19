#include "common.h"
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// Use a bin size equal to cutoff (as in the friend’s implementation)
#define NUM_THREADS 256
#define bin_size cutoff
#define cutoff_squared (cutoff * cutoff)
#define min_r_squared (min_r * min_r)

int blks;
int num_bins;

// We'll use Thrust device_vectors to hold our rebinning arrays.
static thrust::device_vector<int> d_parts_per_bin; // Counts per bin.
static thrust::device_vector<int> d_bin_offsets;   // Exclusive scan over counts.
static thrust::device_vector<int> d_sorted_parts;    // Sorted particle indices (by bin).
static thrust::device_vector<int> d_bin_positions;   // Working copy for assignment.

// ---------------------------------------------------------------------
// Kernel: Compute forces using the sorted particle indices.
// Each thread processes one particle from the sorted array.
__global__ void compute_forces_gpu(particle_t* particles, int num_parts, 
                                   int* sorted_parts, int* bin_offsets, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Get the index of the particle from the sorted order.
    int particle_idx = sorted_parts[tid];
    particle_t& p = particles[particle_idx];

    double ax = 0.0;
    double ay = 0.0;
    double px = p.x;
    double py = p.y;

    // Determine which bin the particle falls into.
    int bin_x = (int)(px / bin_size);
    int bin_y = (int)(py / bin_size);

    // Identify neighbor bins (own bin and the eight neighbors).
    int start_x = max(0, bin_x - 1);
    int end_x = min(num_bins - 1, bin_x + 1);
    int start_y = max(0, bin_y - 1);
    int end_y = min(num_bins - 1, bin_y + 1);

    for (int i = start_x; i <= end_x; i++) {
        for (int j = start_y; j <= end_y; j++) {
            int bin_index = i * num_bins + j;
            // Determine start and end positions for this bin in the sorted array.
            int bin_start = (bin_index == 0) ? 0 : bin_offsets[bin_index - 1];
            int bin_end = bin_offsets[bin_index];
            for (int k = bin_start; k < bin_end; k++) {
                int neighbor_idx = sorted_parts[k];
                if (particle_idx == neighbor_idx)
                    continue;
                particle_t& n = particles[neighbor_idx];
                double dx = n.x - px;
                double dy = n.y - py;
                double r2 = dx * dx + dy * dy;
                if (r2 > cutoff_squared)
                    continue;
                r2 = (r2 > min_r_squared) ? r2 : min_r_squared;
                double r = sqrt(r2);
                double coef = (1 - cutoff / r) / r2 / mass;
                ax += coef * dx;
                ay += coef * dy;
            }
        }
    }
    p.ax = ax;
    p.ay = ay;
}

// ---------------------------------------------------------------------
// Kernel: Move particles using Velocity Verlet integration.
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    // Bounce from walls.
    while (p->x < 0 || p->x > size) {
        p->x = (p->x < 0) ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = (p->y < 0) ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// ---------------------------------------------------------------------
// Kernel: Count the number of particles per bin.
__global__ void compute_parts_per_bin(particle_t* parts, int num_parts, int* parts_per_bin, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    int bin_x = (int)(parts[tid].x / bin_size);
    int bin_y = (int)(parts[tid].y / bin_size);
    int bin_index = bin_x * num_bins + bin_y;
    atomicAdd(&parts_per_bin[bin_index], 1);
}

// ---------------------------------------------------------------------
// Kernel: Assign particles to bins (scatter their indices).
__global__ void assign_parts_to_bins(particle_t* parts, int num_parts, int* sorted_parts, int* bin_positions, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    int bin_x = (int)(parts[tid].x / bin_size);
    int bin_y = (int)(parts[tid].y / bin_size);
    int bin_index = bin_x * num_bins + bin_y;
    int pos = atomicAdd(&bin_positions[bin_index], 1);
    sorted_parts[pos] = tid;
}

// ---------------------------------------------------------------------
// Initialization function.
// (Assumes parts are already in GPU memory.)
void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    num_bins = static_cast<int>(ceil(size / bin_size));

    // Resize Thrust vectors.
    d_parts_per_bin.resize(num_bins * num_bins);
    d_bin_offsets.resize(num_bins * num_bins);
    d_sorted_parts.resize(num_parts);
    d_bin_positions.resize(num_bins * num_bins);
}

// ---------------------------------------------------------------------
// Simulation step: Rebin particles, compute forces, and move particles.
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Step 1: Count particles per bin.
    thrust::fill(d_parts_per_bin.begin(), d_parts_per_bin.end(), 0);
    compute_parts_per_bin<<<blks, NUM_THREADS>>>(parts, num_parts, 
        thrust::raw_pointer_cast(d_parts_per_bin.data()), num_bins);
    cudaDeviceSynchronize();

    // Step 2: Compute bin offsets using an exclusive scan.
    thrust::exclusive_scan(d_parts_per_bin.begin(), d_parts_per_bin.end(), d_bin_offsets.begin());

    // Step 3: Reset bin positions to the bin offsets.
    thrust::copy(d_bin_offsets.begin(), d_bin_offsets.end(), d_bin_positions.begin());

    // Step 4: Scatter particle indices into the sorted_parts array.
    assign_parts_to_bins<<<blks, NUM_THREADS>>>(parts, num_parts, 
        thrust::raw_pointer_cast(d_sorted_parts.data()),
        thrust::raw_pointer_cast(d_bin_positions.data()), num_bins);
    cudaDeviceSynchronize();

    // Step 5: Compute forces using the sorted particle array.
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, 
        thrust::raw_pointer_cast(d_sorted_parts.data()),
        thrust::raw_pointer_cast(d_bin_offsets.data()), num_bins);
    cudaDeviceSynchronize();

    // Step 6: Move particles.
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// (Optional) Finalize simulation and clean up GPU resources.
void finalize_simulation() {
    // This forces proper cleanup of CUDA resources.
    cudaDeviceReset();
}
