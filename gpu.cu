// gpu.cu
#include "common.h"
#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// Use a bin size equal to the cutoff.
#define CELL_SIZE cutoff  
#define cutoff_squared (cutoff * cutoff)
#define min_r_squared (min_r * min_r)
#define NUM_THREADS 256

// Global variables for simulation.
int blks;       // 1D block count (for kernels launched over particles)
int num_bins;   // Number of bins per dimension (computed as ceil(size/cutoff))

// ---------------------------------------------------------------------
// Kernel: Count how many particles fall in each bin.
// Each thread processes one particle.
__global__ void compute_parts_per_bin(particle_t* parts, int num_parts, int* parts_per_bin, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    // Determine bin indices (using CELL_SIZE == cutoff)
    int x = static_cast<int>(parts[tid].x / CELL_SIZE);
    int y = static_cast<int>(parts[tid].y / CELL_SIZE);
    // Clamp to valid range.
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= num_bins) x = num_bins - 1;
    if(y >= num_bins) y = num_bins - 1;
    atomicAdd(&parts_per_bin[x * num_bins + y], 1);
}

// ---------------------------------------------------------------------
// Kernel: Assign particles to bins.
// Each thread processes one particle and uses an atomic counter to determine its sorted position.
__global__ void assign_parts_to_bins(particle_t* parts, int num_parts, int* sorted_parts, int* bin_positions, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    int x = static_cast<int>(parts[tid].x / CELL_SIZE);
    int y = static_cast<int>(parts[tid].y / CELL_SIZE);
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= num_bins) x = num_bins - 1;
    if(y >= num_bins) y = num_bins - 1;
    int bin = x * num_bins + y;
    int pos = atomicAdd(&bin_positions[bin], 1);
    sorted_parts[pos] = tid;
}

// ---------------------------------------------------------------------
// Kernel: Compute forces for each particle using the sorted order and bin boundaries.
// Each thread processes one particle (in sorted order) and accumulates forces locally.
__global__ void compute_forces_gpu(particle_t* particles, int num_parts,
                                   int* sorted_parts, int* bin_offsets, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Get the actual particle index from the sorted list.
    int particle_idx = sorted_parts[tid];
    particle_t p = particles[particle_idx];

    // Local accumulation of force.
    double ax = 0.0, ay = 0.0;
    double px = p.x;
    double py = p.y;

    // Determine which bin this particle falls into.
    int bin_x = static_cast<int>(px / CELL_SIZE);
    int bin_y = static_cast<int>(py / CELL_SIZE);
    if (bin_x < 0) bin_x = 0;
    if (bin_y < 0) bin_y = 0;
    if (bin_x >= num_bins) bin_x = num_bins - 1;
    if (bin_y >= num_bins) bin_y = num_bins - 1;

    // Look at neighboring bins (including its own).
    int start_x = max(0, bin_x - 1);
    int end_x = min(num_bins - 1, bin_x + 1);
    int start_y = max(0, bin_y - 1);
    int end_y = min(num_bins - 1, bin_y + 1);

    for (int i = start_x; i <= end_x; i++) {
        for (int j = start_y; j <= end_y; j++) {
            int bin = i * num_bins + j;
            // Determine the start and end indices for this bin.
            int bin_start = (bin == 0) ? 0 : bin_offsets[bin - 1];
            int bin_end = bin_offsets[bin];
            for (int k = bin_start; k < bin_end; k++) {
                int neighbor_idx = sorted_parts[k];
                if (particle_idx == neighbor_idx) continue;
                particle_t n = particles[neighbor_idx];
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
    // Write the accumulated forces back into global memory.
    particles[particle_idx].ax = ax;
    particles[particle_idx].ay = ay;
}

// ---------------------------------------------------------------------
// Kernel: Move particles using a Velocity Verlet integration step.
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particle_t* p = &particles[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;
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
// Initialization: Called once before simulation begins.
// 'parts' lives in GPU memory.
void init_simulation(particle_t* parts, int num_parts, double size) {
    // Determine 1D block count for particle-based kernels.
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    // Determine number of bins (per dimension) based on simulation size.
    num_bins = static_cast<int>(ceil(size / CELL_SIZE));
}

// ---------------------------------------------------------------------
// Simulation step: Rebin particles, compute forces, and move particles.
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int total_bins = num_bins * num_bins;

    // Use Thrust device_vectors to avoid manual memory management.
    static thrust::device_vector<int> parts_per_bin;
    static thrust::device_vector<int> bin_offsets;
    static thrust::device_vector<int> sorted_parts;
    static thrust::device_vector<int> bin_positions;

    parts_per_bin.resize(total_bins);
    thrust::fill(parts_per_bin.begin(), parts_per_bin.end(), 0);

    // Count particles per bin.
    compute_parts_per_bin<<<blks, NUM_THREADS>>>(parts, num_parts,
        thrust::raw_pointer_cast(parts_per_bin.data()), num_bins);
    cudaDeviceSynchronize();

    // Compute bin offsets via an exclusive scan.
    bin_offsets.resize(total_bins);
    thrust::exclusive_scan(parts_per_bin.begin(), parts_per_bin.end(), bin_offsets.begin());

    // Create a working copy for assigning particles.
    bin_positions.resize(total_bins);
    thrust::copy(bin_offsets.begin(), bin_offsets.end(), bin_positions.begin());

    // Resize the sorted_parts array to hold all particle indices.
    sorted_parts.resize(num_parts);

    // *** DO NOT re-run compute_parts_per_bin here ***
    // Directly assign particles to bins using the previously computed counts.
    assign_parts_to_bins<<<blks, NUM_THREADS>>>(parts, num_parts,
        thrust::raw_pointer_cast(sorted_parts.data()),
        thrust::raw_pointer_cast(bin_positions.data()), num_bins);
    cudaDeviceSynchronize();

    // Compute forces using the sorted order and bin boundaries.
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts,
        thrust::raw_pointer_cast(sorted_parts.data()),
        thrust::raw_pointer_cast(bin_offsets.data()), num_bins);
    cudaDeviceSynchronize();

    // Move particles.
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
}
