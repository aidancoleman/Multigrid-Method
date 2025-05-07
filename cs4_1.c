#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//GLOBAL VARIABLES THAT CAN BE TUNED
static const double OMEGA =  2.0/3.0; /* smoothing weight: omega = 2/3 gives smoothing factor of about 1/3 */
static const int NU =  2; /* # of pre- and post-smoothing Jacobi sweeps */
static const int MAX_CYCLES= 50; /* max # of full V-cycles */
static const double TOL = 1e-8; /* convergence tolerance on L2 norm of residual */

//forward declares

void smooth_jacobi(int l);
void restrict_full(int lf);
void prolongate_and_correct(int lc);
void compute_residual(int l);
void vcycle(int l);
void allocate_mgrid(void);
void free_mem(void);
double residual_norm(int l);

/*Initialise grid sizes at each level, number of grid points will be 2^(level number) + 1, so finer grids (higher levels) have more grid points and coarser have less, e.g. Level 1: 2^(1) + 1 = 3 grid points = one interior grid point in the coarsest grid to solve directly*/
static const int num_levels = 5; //set 5 mulitgrid levels to start
static int N[num_levels]; //array storing number of grid points at each coarseness level

//fill up sizes and create matrices with results for each grid size
static double *u[num_levels], *f[num_levels], *res[num_levels], *tmp[num_levels]; //tmp is for temporary storage of Weighted Jacobi Updates, res[i] is the residual on level i, f[i] = rhs on level i, u[i] = solution at current iteration on level i
/* allocate arrays */
void allocate_mgrid(void){
    for (int i = 0; i<num_levels; ++i){
        N[i] = (1 << (i+1)) + 1;
        int size = N[i]*N[i];
        u[i] = calloc(size, sizeof(double)); //set them to zero to start
        f[i] = calloc(size, sizeof(double));
        res[i] = calloc(size, sizeof(double));
        tmp[i] = calloc(size, sizeof(double));
        if(!u[i]||!f[i]||!res[i]||!tmp[i]){
            fprintf(stderr,"Allocation failed at level %d\n",i);
            exit(1);
        }
    }
}
#define N0  N[0] /*coarsest*/
   
//helper for cleaning up all the arrays I just calloc'd and malloc'd
void free_mem()
{
    for(int i=0; i<num_levels; i++){
        free(u[i]);
        free(f[i]);
        free(res[i]);
        free(tmp[i]);
    }
}
//macro helper for flattening from 2D to 1D
#define  IDX(l,i,j)  ((i)*(N[l]) + (j))

/*ERROR/RESIDUAL CALCULATION FUNCTIONS*/

/*
  Compute residual:  res = f - Au
  with Au at (i,j) = (4 u_ij - u_{i+/-1,j} - u_{i,j+/-1})*/
void compute_residual(int l)
{
    int n = N[l];
    /* interior points only */
    for(int i=1; i<n-1; i++){
        for(int j=1; j<n-1; j++){
            double Au_ij =
                4.0*u[l][IDX(l,i,j)]
              -   u[l][IDX(l,i-1,j)]
              -   u[l][IDX(l,i+1,j)]
              -   u[l][IDX(l,i,j-1)]
              -   u[l][IDX(l,i,j+1)];
            res[l][IDX(l,i,j)] = f[l][IDX(l,i,j)] - Au_ij;
        }
    }
}


//L2 norm of residual on level p (interior points)

double residual_norm(int p)
{
    int n = N[p]; //n is the size of the level
    double sum = 0.0;
    for(int i=1; i<n-1; i++){
        for(int j=1; j<n-1; j++){
            double r = res[p][IDX(p,i,j)]; //flatten out element p of res matrix corresponding to pth level
            sum += r*r;
        }
    }
    return sqrt(sum);
}
/*WEIGHTED JACOBI SMOOTHING (v sweeps): u_new = (1 - omega)*u_old + (1/4)*(omega)*(sum of 4 neighbours in the 5 point stencil)
 omega: The smoothing weight, 1 - omega = smoothing factor
 NB : If omega is too small, we barely move each sweep => slow smoothing
If omega is too large, some error‐modes actually get amplified rather than dampened => divergence of the smoother.
 
 In a multigrid context we don’t care so much about driving the overall spectral radius as low as possible (for convergence, that’s the coarse‐grid’s job!). We care more about eliminating the high–frequency (oscillatory) components aggressively, so that the coarse‐grid correction can mop up the remaining smooth error.
 
 v: Pre-smoothing: sweeps before you compute the residual and restrict to the next coarser grid.

 Post-smoothing: v sweeps after we prolongate the coarse-grid correction back to the fine grid.

 Too few sweeps (small v):
 The smoother won’t sufficiently kill high-frequency error before we do coarse‐grid correction, so the correction is less effective and need more V-cycles for convergence

 Too many sweeps (large v):
  spend extra work on the fine grid for diminishing returns. Past v = 2 or 3, each additional sweep only wrings out a little more high-freq error, while we could be on the next V-cycle’s coarse‐grid solve
 */
void smooth_jacobi(int l)
{
    int n = N[l];
    double omega = OMEGA;
    for(int sweep=0; sweep<NU; sweep++){
        for(int i=1; i<n-1; i++){
            for(int j=1; j<n-1; j++){
                double up = u[l][IDX(l,i-1,j)];
                double dn = u[l][IDX(l,i+1,j)];
                double lf = u[l][IDX(l,i,j-1)];
                double rt = u[l][IDX(l,i,j+1)];
                double rhs = f[l][IDX(l,i,j)];
                double update = 0.25*(up + dn + lf + rt + rhs);
                tmp[l][IDX(l,i,j)]
                  = (1.0-omega)*u[l][IDX(l,i,j)]
                  +  omega*update;
            }
        }
        /* copy tmp into u (only interior pts) */
        for(int i=1; i<n-1; i++){
            for(int j=1; j<n-1; j++){
                u[l][IDX(l,i,j)] = tmp[l][IDX(l,i,j)];
            }
        }
    }
}
/*
  Full‐weighting restriction:moving data down from the finer grid to coarse grid
    f_coarse[I,J] = (1/16)* sum_{|p|,|q| \leq 1} w(p,q) * res_fine[2I+p,2J+q]
    weights w = [1 2 1; 2 4 2; 1 2 1]
*/
void restrict_full(int l)
{
    int nf     = N[l];
    int coarse = l - 1;
    int nc     = N[coarse];

    /* zero out coarse RHS including boundary */
    for(int i = 0; i < nc; i++)
        for(int j = 0; j < nc; j++)
            f[coarse][i*nc + j] = 0.0;

    /* interior of coarse grid: I=1..nc-2 maps from i=2*I,j=2*J */
    for(int I = 1; I < nc - 1; I++){
        for(int J = 1; J < nc - 1; J++){
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

/*
  Linear‐interpolation prolongation + correction: transfer correction
  from coarse level `l` up to fine level `l+1`.
*/
void prolongate_and_correct(int l)
{
    int nc     = N[l];
    int fine   = l + 1;
    int nf     = N[fine];

    /* inject and interpolate each coarse‐grid value into fine grid */
    for(int I = 0; I < nc; I++){
        for(int J = 0; J < nc; J++){
            double uc = u[l][I*nc + J];
            int i = 2*I, j = 2*J;

            /* direct injection */
            u[fine][i*nf + j] += uc;

            /* east‐west neighbors */
            if(i+1 < nf){
                double ur = (I+1 < nc ? u[l][(I+1)*nc + J] : 0.0);
                u[fine][(i+1)*nf + j] += 0.5*(uc + ur);
            }
            /* north‐south neighbors */
            if(j+1 < nf){
                double uu = (J+1 < nc ? u[l][I*nc + (J+1)] : 0.0);
                u[fine][i*nf + (j+1)] += 0.5*(uc + uu);
            }
            /* diagonal */
            if(i+1 < nf && j+1 < nf){
                double ur  = (I+1 < nc ? u[l][(I+1)*nc + J] : 0.0);
                double uu  = (J+1 < nc ? u[l][I*nc + (J+1)] : 0.0);
                double uru = (I+1 < nc && J+1 < nc ? u[l][(I+1)*nc + (J+1)] : 0.0);
                u[fine][(i+1)*nf + (j+1)] += 0.25*(uc + ur + uu + uru);
            }
        }
    }
}
/*----------------------------------------
 ACCORDING TO ALGO
  Use Recursion for  V-cycle: solve at level 'l'
    1) v pre-smooth
    2) get resid => restrict to l+1
    3) if at coarsest: direct solve, else recurse
    4) prolongate & correct
    5) v post-smooth
----------------------------------------*/
void vcycle(int l) {
  // pre-smooth on level l
  smooth_jacobi(l);

  // compute residual on level l
  compute_residual(l);

  if (l == 0) {
    // coarsest (3×3) direct solve  => only one interior at (1,1)
    u[0][IDX(0,1,1)] = f[0][IDX(0,1,1)] / 4.0;
  }
  else {
    // restrict residual from fine l to coarse l-1
    restrict_full(l);

    // zero‐initialise coarse correction
    for(int i = 0, sz = N[l-1]*N[l-1]; i < sz; i++)
      u[l-1][i] = 0.0;

    // recurse downward
    vcycle(l-1);

    // prolongate & add correction back to level l
    prolongate_and_correct(l-1);
  }

  // post-smooth on level l
  smooth_jacobi(l);
}
/*----------------------------------------
  Main
----------------------------------------*/
int main(void){
        allocate_mgrid();
        printf("Multigrid Poisson solver: finest N=%d, levels=%d, ω=%g, ν=%d\n",
               N[num_levels-1], num_levels, OMEGA, NU);
    
        /* initialize RHS on the finest level = h^2 * f(x,y), f(x,y)=1 */
        {
          int L = num_levels-1;
          int n = N[L];
          double h  = 1.0/(n-1),
                 h2 = h*h;
          for(int i=1; i<n-1; i++){
            for(int j=1; j<n-1; j++){
              f[L][IDX(L,i,j)] = 1.0 * h2;
            }
          }
          // u[L] is already zero from calloc
        }

        /* compute initial residual on finest grid */
        compute_residual(num_levels-1);
        double res0 = residual_norm(num_levels-1);
        printf("Initial residual norm: %g\n", res0);

        /* V-cycles, descending from finest (L) down to coarsest (0) */
        for(int cycle=1; cycle<=MAX_CYCLES; cycle++){
          vcycle(num_levels-1);
          compute_residual(num_levels-1);
          double resn = residual_norm(num_levels-1);
          printf(" V-cycle %2d → residual norm = %g\n", cycle, resn);

          if(resn < TOL){
            printf(" Converged in %d V-cycles (||res|| < %g)\n", cycle, TOL);
            break;
          }
            /* simple divergence check, make sure we are not wastefully going through iterations while modes are diverging */
          if(resn > 1e4 * res0){
            fprintf(stderr, " Divergence detected. Aborting.\n");
            break;
          }
          if(cycle == MAX_CYCLES){
            printf(" Reached maximum V-cycles (%d) without full convergence.\n",
                   MAX_CYCLES);
          }
        }

        free_mem();
        return 0;
    }
      
