#include "common.h"
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define NUM_THREADS 256
#define CELL_SIZE (1.2 * cutoff)  // cell size is slightly larger than the cutoff

// Global variables used across simulation steps and kernels.
int blks;  // Number of CUDA blocks for 1D kernels (used in rebinning)

// Cell grid globals for rebinning.
int num_cells_x, num_cells_y;
int* d_cell_starts;    // first index for each cell
int* d_cell_ends;      // one past the last index for each cell
int* d_cell_counts;    // count of particles per cell (used in rebinning)
int* d_cell_offsets;   // working copy of cell_starts for scattering

// Temporary sorted particle array (by cell).
particle_t* d_particles_sorted;

// Ghost particles and ghost count.
// NOTE: Instead of an array of ghost counts (indexed per block), we now use a single counter.
particle_t* d_ghost_particles;
int* d_ghost_count;  // single integer (allocated as a device pointer)

// ---------------------------------------------------------------------
// Device function: Compute and apply force between two particles.
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// ---------------------------------------------------------------------
// Kernel: Compute forces between particles in each cell and the 8 neighboring cells.
__global__ void compute_forces_gpu(particle_t* particles, int num_parts,
                                   int* cell_starts, int* cell_ends,
                                   int num_cells_x, int num_cells_y) {
    // Use 2D grid: each thread corresponds to a cell (by its x,y indices)
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if (tx >= num_cells_x || ty >= num_cells_y)
        return;
    
    int cell_index = tx + ty * num_cells_x;
    int start = cell_starts[cell_index];
    int end = cell_ends[cell_index];

    // Loop over this cell and its 8 neighbors.
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int neighbor_x = tx + dx;
            int neighbor_y = ty + dy;
            if (neighbor_x < 0 || neighbor_x >= num_cells_x ||
                neighbor_y < 0 || neighbor_y >= num_cells_y)
                continue;
            int neighbor_index = neighbor_x + neighbor_y * num_cells_x;
            int neighbor_start = cell_starts[neighbor_index];
            int neighbor_end = cell_ends[neighbor_index];
            for (int i = start; i < end; i++) {
                for (int j = neighbor_start; j < neighbor_end; j++) {
                    if (i != j) {  // avoid self-interaction
                        apply_force_gpu(particles[i], particles[j]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Kernel: Compute additional forces using ghost particles.
__global__ void compute_forces_with_ghosts(particle_t* parts, int num_parts,
                                             particle_t* ghost_particles,
                                             int* ghost_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    particle_t& p = parts[tid];
    // Use the global ghost count.
    int count = *ghost_count;
    for (int i = 0; i < count; i++) {
        apply_force_gpu(p, ghost_particles[i]);
    }
}

// ---------------------------------------------------------------------
// Kernel: Move particles using Velocity Verlet integration.
// If a particle leaves the domain, record it as a ghost particle.
__global__ void move_gpu(particle_t* particles, int num_parts, double size,
                           particle_t* ghost_particles, int* ghost_count) {
    // Compute a unique thread id from 2D grid.
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tx + ty * gridDim.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    particle_t* p = &particles[tid];
    // Velocity Verlet integration update.
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

    // If still outside the domain, record as a ghost particle.
    if (p->x < 0 || p->x > size || p->y < 0 || p->y > size) {
        int idx = atomicAdd(ghost_count, 1);
        ghost_particles[idx] = *p;
    }
}

// ---------------------------------------------------------------------
// Rebinning Kernels

// Kernel: Clear the cell count array.
__global__ void clear_cell_counts(int* cell_counts, int total_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells)
        cell_counts[idx] = 0;
}

// Kernel: Count particles per cell.
__global__ void count_particles_kernel(particle_t* particles, int num_parts,
                                         int num_cells_x, int num_cells_y,
                                         double cell_size, int* cell_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        particle_t p = particles[idx];
        int cell_x = (int)(p.x / cell_size);
        int cell_y = (int)(p.y / cell_size);
        // Clamp indices.
        cell_x = (cell_x < 0) ? 0 : (cell_x >= num_cells_x ? num_cells_x - 1 : cell_x);
        cell_y = (cell_y < 0) ? 0 : (cell_y >= num_cells_y ? num_cells_y - 1 : cell_y);
        int cell_id = cell_x + cell_y * num_cells_x;
        atomicAdd(&cell_counts[cell
