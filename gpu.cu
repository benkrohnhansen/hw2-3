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

// Ghost particles and ghost counts.
// (Note: ghost_counts now is allocated based on the x-dimension of the 2D grid 
// used in move_gpu and ghost-force kernels so that indexing (blockIdx.x) is valid.)
particle_t* d_ghost_particles;
int* d_ghost_counts;

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
                                             int* ghost_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    particle_t& p = parts[tid];
    // Use ghost particles from the current block (if any).
    int count = ghost_counts[blockIdx.x];  // note: only blockIdx.x is used
    for (int i = 0; i < count; i++) {
        apply_force_gpu(p, ghost_particles[i]);
    }
}

// ---------------------------------------------------------------------
// Kernel: Move particles using Velocity Verlet integration.
// If a particle leaves the domain, record it as a ghost particle.
__global__ void move_gpu(particle_t* particles, int num_parts, double size,
                           particle_t* ghost_particles, int* ghost_counts) {
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
        int idx = atomicAdd(&ghost_counts[blockIdx.x], 1);
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
        atomicAdd(&cell_counts[cell_id], 1);
    }
}

// Kernel: Scatter particles into a sorted array (sorted by cell).
__global__ void scatter_particles_kernel(particle_t* particles, int num_parts,
                                           int num_cells_x, int num_cells_y,
                                           double cell_size, int* cell_offsets,
                                           particle_t* particles_sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        particle_t p = particles[idx];
        int cell_x = (int)(p.x / cell_size);
        int cell_y = (int)(p.y / cell_size);
        cell_x = (cell_x < 0) ? 0 : (cell_x >= num_cells_x ? num_cells_x - 1 : cell_x);
        cell_y = (cell_y < 0) ? 0 : (cell_y >= num_cells_y ? num_cells_y - 1 : cell_y);
        int cell_id = cell_x + cell_y * num_cells_x;
        int pos = atomicAdd(&cell_offsets[cell_id], 1);
        particles_sorted[pos] = p;
    }
}

// Kernel: Update cell_ends based on cell_starts and counts.
__global__ void update_cell_ends_kernel(int total_cells,
                                          int* cell_starts,
                                          int* cell_counts,
                                          int* cell_ends) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells)
        cell_ends[idx] = cell_starts[idx] + cell_counts[idx];
}

// ---------------------------------------------------------------------
// Rebinning host function.
// This function clears counts, counts particles per cell,
// computes an exclusive scan (to get cell_starts), scatters particles,
// updates cell_ends, and then copies the sorted array back.
void rebin_particles(particle_t* particles, int num_parts, double cell_size,
                     int num_cells_x, int num_cells_y) {
    int total_cells = num_cells_x * num_cells_y;
    int threads = 256;
    int blocks = (total_cells + threads - 1) / threads;

    // Clear the cell_counts array.
    clear_cell_counts<<<blocks, threads>>>(d_cell_counts, total_cells);
    cudaDeviceSynchronize();
    
    // Count particles in each cell.
    blocks = (num_parts + threads - 1) / threads;
    count_particles_kernel<<<blocks, threads>>>(particles, num_parts,
                                                  num_cells_x, num_cells_y,
                                                  cell_size, d_cell_counts);
    cudaDeviceSynchronize();
    
    // Perform an exclusive scan on cell_counts to compute cell_starts.
    thrust::device_ptr<int> counts_ptr(d_cell_counts);
    thrust::device_ptr<int> starts_ptr(d_cell_starts);
    thrust::exclusive_scan(counts_ptr, counts_ptr + total_cells, starts_ptr);
    
    // Copy cell_starts into a working array for scattering.
    cudaMemcpy(d_cell_offsets, d_cell_starts, total_cells * sizeof(int),
               cudaMemcpyDeviceToDevice);
    
    // Scatter particles into the sorted array.
    blocks = (num_parts + threads - 1) / threads;
    scatter_particles_kernel<<<blocks, threads>>>(particles, num_parts,
                                                    num_cells_x, num_cells_y,
                                                    cell_size, d_cell_offsets,
                                                    d_particles_sorted);
    cudaDeviceSynchronize();
    
    // Update cell_ends.
    blocks = (total_cells + threads - 1) / threads;
    update_cell_ends_kernel<<<blocks, threads>>>(total_cells, d_cell_starts,
                                                 d_cell_counts, d_cell_ends);
    cudaDeviceSynchronize();
    
    // Copy the sorted particles back into the main array.
    cudaMemcpy(particles, d_particles_sorted, num_parts * sizeof(particle_t),
               cudaMemcpyDeviceToDevice);
}

// ---------------------------------------------------------------------
// Initialization function.
// This function is called once (before simulation begins) and sets up the
// cell grid arrays as well as ghost-particle storage.
void init_simulation(particle_t* parts, int num_parts, double size) {
    // Determine the number of blocks for 1D kernels.
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    
    // Set up the cell grid dimensions.
    num_cells_x = (int)(size / CELL_SIZE);
    num_cells_y = (int)(size / CELL_SIZE);
    int total_cells = num_cells_x * num_cells_y;
    
    // Allocate memory for cell arrays.
    cudaMalloc((void**)&d_cell_starts, total_cells * sizeof(int));
    cudaMalloc((void**)&d_cell_ends, total_cells * sizeof(int));
    cudaMalloc((void**)&d_cell_counts, total_cells * sizeof(int));
    cudaMalloc((void**)&d_cell_offsets, total_cells * sizeof(int));
    cudaMalloc((void**)&d_particles_sorted, num_parts * sizeof(particle_t));
    
    // Allocate ghost particle arrays.
    cudaMalloc((void**)&d_ghost_particles, num_parts * sizeof(particle_t));
    // Allocate ghost_counts based on the x-dimension of the grid we will use
    int ghost_num_blocks = (num_cells_x + 16 - 1) / 16;
    cudaMalloc((void**)&d_ghost_counts, ghost_num_blocks * sizeof(int));
    cudaMemset(d_ghost_counts, 0, ghost_num_blocks * sizeof(int));
    
    // Initialize cell_counts to zero.
    cudaMemset(d_cell_counts, 0, total_cells * sizeof(int));
}

// ---------------------------------------------------------------------
// Simulation step: rebin particles, compute forces, move particles, and compute ghost forces.
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Rebin particles: update cell arrays based on current particle positions.
    rebin_particles(parts, num_parts, CELL_SIZE, num_cells_x, num_cells_y);
    
    // Set up a 2D grid for the force and move kernels.
    dim3 blockDim(16, 16);
    dim3 gridDim((num_cells_x + blockDim.x - 1) / blockDim.x,
                 (num_cells_y + blockDim.y - 1) / blockDim.y);
    
    // Compute forces among particles using the sorted array and cell arrays.
    compute_forces_gpu<<<gridDim, blockDim>>>(parts, num_parts, d_cell_starts,
                                                d_cell_ends, num_cells_x, num_cells_y);
    cudaDeviceSynchronize();
    
    // Reset ghost_counts for the current simulation step.
    int ghost_num_blocks = (num_cells_x + 16 - 1) / 16;
    cudaMemset(d_ghost_counts, 0, ghost_num_blocks * sizeof(int));
    
    // Move particles (and record any that still lie outside the domain).
    move_gpu<<<gridDim, blockDim>>>(parts, num_parts, size,
                                    d_ghost_particles, d_ghost_counts);
    cudaDeviceSynchronize();
    
    // Compute additional forces from ghost particles.
    compute_forces_with_ghosts<<<gridDim, blockDim>>>(parts, num_parts,
                                                      d_ghost_particles, d_ghost_counts);
    cudaDeviceSynchronize();
}
