#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// tunable solver parameters
static const double OMEGA      = 2.0/3.0;
static const int    NU         = 2;
static const int    MAX_CYCLES = 1500;
static const double TOL        = 1e-7;

// grid‐hierarchy (set at run‐time)
int    num_levels;    // current l_max
int    Lfinest;       // = num_levels-1
int    Lcoarse = 0;   // coarsest level = 0
int   *N;             // N[l] = # grid‐points per dimension at level l
double **u, **f, **res, **tmp;

// instrumentation
long   coarse_solves;
double *res_hist;

// flatten 2D index
#define IDX(l,i,j)  ((i)*N[l] + (j))

// forward declarations
void setup_grids(int Nsub);
void free_mem(void);
void compute_residual(int l);
double residual_norm(int l);
void smooth_jacobi(int l);
void restrict_full(int l);
void prolongate_and_correct(int l);
void vcycle(int l);
int  compute_max_levels_for_coarse8(int Nsub);
void run_experiment(int Nsub, int levels);
// Compute how many levels so that coarse N[0] ≤ 8
int compute_max_levels_for_coarse8(int Nsub) {
    // temporary array just to track coarsening
   int Nt[32];
    int levels = 1;
    Nt[levels] = Nsub + 1;
    while (levels < 31) {
        levels++;
        Nt[levels] = (Nt[levels-1] - 1)/2 + 1;
        if (Nt[levels] <= 8) break;
    }
    return levels + 1;  // +1 because levels=1 means levels=2, etc.
}

// Run one MG solve on grid-size Nsub with levels levels.
// Prints: Nsub,setup,levels,cycles,time,final_residual
void run_experiment(int Nsub, int levels) {
    num_levels = levels;
    Lfinest = num_levels - 1;
    Lcoarse = 0;

    // --- allocate arrays just as before, using Nsub,num_levels ---
    N   = malloc(num_levels * sizeof *N);
    u   = malloc(num_levels * sizeof *u);
    f   = malloc(num_levels * sizeof *f);
    res = malloc(num_levels * sizeof *res);
    tmp = malloc(num_levels * sizeof *tmp);
    res_hist = malloc((MAX_CYCLES+2)*sizeof *res_hist);

    setup_grids(Nsub);
    // initialize f on finest level (unchanged) …
    { int ℓ = Lfinest, n = N[ℓ];
      double h = 1.0/(n-1), h2 = h*h;
      for(int i=1;i<n-1;i++) for(int j=1;j<n-1;j++){
        double x=i*h, y=j*h;
        f[ℓ][IDX(ℓ,i,j)] = 2*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y)*h2;
      }
    }

    // --- run V‐cycles ---
    coarse_solves = 0;
    clock_t t0 = clock();
    compute_residual(Lfinest);
    res_hist[0]=residual_norm(Lfinest);
    int cycle;
    for(cycle=1; cycle<=MAX_CYCLES; ++cycle){
      vcycle(Lfinest);
      compute_residual(Lfinest);
      res_hist[cycle]=residual_norm(Lfinest);
      if(res_hist[cycle]<TOL) break;
    }
    double runtime = (clock()-t0)/(double)CLOCKS_PER_SEC;

    // final residual
    int last = (cycle <= MAX_CYCLES ? cycle : MAX_CYCLES);
        double rfinal = res_hist[last];

    // CSV output
    printf("%4d,%s,%2d,%4d,%.4f,%.3e\n",
           Nsub,
           (num_levels==2?"2-lv":"max"),
           num_levels,
           cycle,
           runtime,
           rfinal);

    // cleanup
    free_mem();
    free(res_hist);
    free(N); free(u); free(f); free(res); free(tmp);
}

int main(void) {
    // header for CSV
    puts("Nsub,setup,num_levels,cycles,time_s,final_res");

    int grid[] = {16,32,64,128,256};
    for(int i=0;i<5;i++){
        int Nsub = grid[i];
        // 1) two-level MG
        run_experiment(Nsub, 2);
        // 2) maximal MG down to N[0]=8
        int maxlev = compute_max_levels_for_coarse8(Nsub);
        run_experiment(Nsub, maxlev);
    }
    return 0;
}
// build level sizes and calloc u,f,res,tmp at each level
void setup_grids(int Nsub){
    // finest level has Nsub+1 points
    N[Lfinest] = Nsub + 1;
    // each coarser halves the interior
    for(int ℓ = Lfinest-1; ℓ >= 0; --ℓ){
        N[ℓ] = (N[ℓ+1] - 1)/2 + 1;
    }
    // allocate and zero each level
    for(int ℓ = 0; ℓ < num_levels; ++ℓ){
        int sz = N[ℓ]*N[ℓ];
        u[ℓ]   = calloc(sz, sizeof(double));
        f[ℓ]   = calloc(sz, sizeof(double));
        res[ℓ] = calloc(sz, sizeof(double));
        tmp[ℓ] = calloc(sz, sizeof(double));
        if(!u[ℓ]||!f[ℓ]||!res[ℓ]||!tmp[ℓ]){
            fprintf(stderr,"alloc failed at level %d\n",ℓ);
            exit(1);
        }
    }
}

void free_mem(void){
    for(int ℓ = 0; ℓ < num_levels; ++ℓ){
        free(u[ℓ]);
        free(f[ℓ]);
        free(res[ℓ]);
        free(tmp[ℓ]);
    }
}

// compute residual:  r = f – A u
void compute_residual(int l){
    int n = N[l];
    for(int i = 1; i < n-1; ++i){
        for(int j = 1; j < n-1; ++j){
            double Au = 4.0*u[l][IDX(l,i,j)]
                      -   u[l][IDX(l,i-1,j)]
                      -   u[l][IDX(l,i+1,j)]
                      -   u[l][IDX(l,i,j-1)]
                      -   u[l][IDX(l,i,j+1)];
            res[l][IDX(l,i,j)] = f[l][IDX(l,i,j)] - Au;
        }
    }
}

double residual_norm(int l){
    int n = N[l];
    double sum = 0.0;
    for(int i = 1; i < n-1; ++i){
        for(int j = 1; j < n-1; ++j){
            double r = res[l][IDX(l,i,j)];
            sum += r*r;
        }
    }
    return sqrt(sum);
}

void smooth_jacobi(int l){
    int n = N[l];
    for(int sweep = 0; sweep < NU; ++sweep){
        for(int i = 1; i < n-1; ++i){
            for(int j = 1; j < n-1; ++j){
                double up  = u[l][IDX(l,i-1,j)];
                double dn  = u[l][IDX(l,i+1,j)];
                double lf  = u[l][IDX(l,i,j-1)];
                double rt  = u[l][IDX(l,i,j+1)];
                double rhs = f[l][IDX(l,i,j)];
                double upd = 0.25*(up + dn + lf + rt + rhs);
                tmp[l][IDX(l,i,j)] = (1.0-OMEGA)*u[l][IDX(l,i,j)]
                                   + OMEGA*upd;
            }
        }
        // copy back
        for(int i = 1; i < n-1; ++i)
            for(int j = 1; j < n-1; ++j)
                u[l][IDX(l,i,j)] = tmp[l][IDX(l,i,j)];
    }
}

void restrict_full(int l){
    int nf     = N[l];
    int coarse = l - 1;
    int nc     = N[coarse];
    // zero coarse RHS
    for(int i = 0; i < nc; ++i)
        for(int j = 0; j < nc; ++j)
            f[coarse][i*nc + j] = 0.0;
    // full‐weighting
    for(int I = 1; I < nc-1; ++I){
        for(int J = 1; J < nc-1; ++J){
            int i = 2*I, j = 2*J;
            double sum =
                 res[l][(i-1)*nf + (j-1)]
               + 2*res[l][(i  )*nf + (j-1)]
               +    res[l][(i+1)*nf + (j-1)]
               + 2*res[l][(i-1)*nf + (j  )]
               + 4*res[l][(i  )*nf + (j  )]
               + 2*res[l][(i+1)*nf + (j  )]
               +    res[l][(i-1)*nf + (j+1)]
               + 2*res[l][(i  )*nf + (j+1)]
               +    res[l][(i+1)*nf + (j+1)];
            f[coarse][I*nc + J] = sum * (1.0/16.0);
        }
    }
}

void prolongate_and_correct(int l){
    int nc   = N[l];
    int fine = l + 1;
    int nf   = N[fine];
    for(int I = 0; I < nc; ++I){
        for(int J = 0; J < nc; ++J){
            double uc = u[l][I*nc + J];
            int i = 2*I, j = 2*J;
            u[fine][i*nf + j] += uc;
            if(i+1 < nf){
                double ur = (I+1<nc?u[l][(I+1)*nc+J]:0.0);
                u[fine][(i+1)*nf + j] += 0.5*(uc + ur);
            }
            if(j+1 < nf){
                double uu = (J+1<nc?u[l][I*nc+(J+1)]:0.0);
                u[fine][i*nf + (j+1)] += 0.5*(uc + uu);
            }
            if(i+1<nf && j+1<nf){
                double ur  = (I+1<nc?u[l][(I+1)*nc+J]:0.0);
                double uu  = (J+1<nc?u[l][I*nc+(J+1)]:0.0);
                double uru = (I+1<nc&&J+1<nc?u[l][(I+1)*nc+(J+1)]:0.0);
                u[fine][(i+1)*nf + (j+1)]
                  += 0.25*(uc + ur + uu + uru);
            }
        }
    }
}

void vcycle(int l){
    smooth_jacobi(l);
    compute_residual(l);

    if(l == Lcoarse){
        coarse_solves++;
        // direct solve on 1×1
        u[l][IDX(l,1,1)] = f[l][IDX(l,1,1)] / 4.0;
    } else {
        restrict_full(l);
        // zero‐init coarse solution
        int csz = N[l-1]*N[l-1];
        for(int i = 0; i < csz; ++i) u[l-1][i] = 0.0;
        vcycle(l-1);
        prolongate_and_correct(l-1);
    }

    smooth_jacobi(l);
}


