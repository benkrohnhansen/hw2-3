#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
} particle_t;

extern int num_cells_x, num_cells_y;
extern int* d_cell_starts;
extern int* d_cell_ends;
extern particle_t* d_ghost_particles;
extern int* d_ghost_counts;



// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size);
extern __global__ void compute_forces_with_ghosts(particle_t* parts, int num_parts, particle_t* ghost_particles, int* ghost_counts);


#endif
