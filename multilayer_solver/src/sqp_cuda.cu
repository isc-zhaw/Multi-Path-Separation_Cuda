/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sqp_cuda.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 09-Apr-2024 10:22:26
 */

/* Include Files */
#include <math.h>
#include <string.h>

#include "private/helper_sqp.cuh"
#include "private/sqp_cuda.cuh"
#include "private/sqp_cuda_types.cuh"

__device__ const int kMaxIter = 500;

/* Type Definitions */
#ifndef typedef_struct_T
#define typedef_struct_T

typedef struct {
  float pi4fvek_c[5];
  float mmvek[10];
  bool valid_freq[5];
} struct_T;

#endif /* typedef_struct_T */

#ifndef typedef_anonymous_function
#define typedef_anonymous_function

typedef struct {
  struct_T workspace;
} anonymous_function;

#endif /* typedef_anonymous_function */

#ifndef c_typedef_coder_internal_sticky
#define c_typedef_coder_internal_sticky

typedef struct {
  anonymous_function value;
} coder_internal_stickyStruct;

#endif /* c_typedef_coder_internal_sticky */

#ifndef c_typedef_b_coder_internal_stic
#define c_typedef_b_coder_internal_stic

typedef struct {
  coder_internal_stickyStruct next;
} b_coder_internal_stickyStruct;

#endif /* c_typedef_b_coder_internal_stic */

#ifndef c_typedef_c_coder_internal_stic
#define c_typedef_c_coder_internal_stic

typedef struct {
  b_coder_internal_stickyStruct next;
} c_coder_internal_stickyStruct;

#endif /* c_typedef_c_coder_internal_stic */

#ifndef c_typedef_d_coder_internal_stic
#define c_typedef_d_coder_internal_stic

typedef struct {
  c_coder_internal_stickyStruct next;
} d_coder_internal_stickyStruct;

#endif /* c_typedef_d_coder_internal_stic */

#ifndef c_typedef_e_coder_internal_stic
#define c_typedef_e_coder_internal_stic

typedef struct {
  d_coder_internal_stickyStruct next;
} e_coder_internal_stickyStruct;

#endif /* c_typedef_e_coder_internal_stic */

#ifndef c_typedef_f_coder_internal_stic
#define c_typedef_f_coder_internal_stic

typedef struct {
  e_coder_internal_stickyStruct next;
} f_coder_internal_stickyStruct;

#endif /* c_typedef_f_coder_internal_stic */

#ifndef c_typedef_g_coder_internal_stic
#define c_typedef_g_coder_internal_stic

typedef struct {
  f_coder_internal_stickyStruct next;
} g_coder_internal_stickyStruct;

#endif /* c_typedef_g_coder_internal_stic */

#ifndef c_typedef_h_coder_internal_stic
#define c_typedef_h_coder_internal_stic

typedef struct {
  g_coder_internal_stickyStruct next;
} h_coder_internal_stickyStruct;

#endif /* c_typedef_h_coder_internal_stic */

#ifndef c_typedef_i_coder_internal_stic
#define c_typedef_i_coder_internal_stic

typedef struct {
  h_coder_internal_stickyStruct next;
} i_coder_internal_stickyStruct;

#endif /* c_typedef_i_coder_internal_stic */

#ifndef typedef_b_struct_T
#define typedef_b_struct_T

typedef struct {
  float workspace_float[45];
  int workspace_int[9];
  int workspace_sort[9];
} b_struct_T;

#endif /* typedef_b_struct_T */

#ifndef typedef_c_struct_T
#define typedef_c_struct_T

typedef struct {
  int ldq;
  float QR[81];
  float Q[81];
  int jpvt[9];
  int mrows;
  int ncols;
  float tau[9];
  int minRowCol;
  bool usedPivoting;
} c_struct_T;

#endif /* typedef_c_struct_T */

#ifndef typedef_d_struct_T
#define typedef_d_struct_T

typedef struct {
  int nVarMax;
  int mNonlinIneq;
  int mNonlinEq;
  int mIneq;
  int mEq;
  int iNonIneq0;
  int iNonEq0;
  float sqpFval;
  float sqpFval_old;
  float xstarsqp[4];
  float xstarsqp_old[4];
  float grad[5];
  float grad_old[5];
  int FunctionEvaluations;
  int sqpIterations;
  int sqpExitFlag;
  float lambdasqp[9];
  float lambdaStopTest[9];
  float lambdaStopTestPrev[9];
  float steplength;
  float delta_x[5];
  float socDirection[5];
  int workingset_old[9];
  float gradLag[5];
  float delta_gradLag[5];
  float xstar[5];
  float fstar;
  float firstorderopt;
  float lambda[9];
  int state;
  float maxConstr;
  int iterations;
  float searchDir[5];
} d_struct_T;

#endif /* typedef_d_struct_T */

#ifndef typedef_e_struct_T
#define typedef_e_struct_T

typedef struct {
  int mConstr;
  int mConstrOrig;
  int mConstrMax;
  int nVar;
  int nVarOrig;
  int nVarMax;
  int ldA;
  float lb[5];
  float ub[5];
  int indexLB[5];
  int indexUB[5];
  int indexFixed[5];
  int mEqRemoved;
  float ATwset[45];
  float bwset[9];
  int nActiveConstr;
  float maxConstrWorkspace[9];
  int sizes[5];
  int sizesNormal[5];
  int sizesPhaseOne[5];
  int sizesRegularized[5];
  int sizesRegPhaseOne[5];
  int isActiveIdx[6];
  int isActiveIdxNormal[6];
  int isActiveIdxPhaseOne[6];
  int isActiveIdxRegularized[6];
  int isActiveIdxRegPhaseOne[6];
  bool isActiveConstr[9];
  int Wid[9];
  int Wlocalidx[9];
  int nWConstr[5];
  int probType;
  double SLACK0;
} e_struct_T;

#endif /* typedef_e_struct_T */

#ifndef typedef_f_struct_T
#define typedef_f_struct_T

typedef struct {
  anonymous_function objfun;
  float f_1;
  float f_2;
  int nVar;
  int numEvals;
  bool SpecifyObjectiveGradient;
  bool hasLB[4];
  bool hasUB[4];
  int FiniteDifferenceType;
} f_struct_T;

#endif /* typedef_f_struct_T */

#ifndef typedef_g_struct_T
#define typedef_g_struct_T

typedef struct {
  char SolverName[7];
  int MaxIterations;
  float StepTolerance;
  float ConstraintTolerance;
  float ObjectiveLimit;
  float PricingTolerance;
  float ConstrRelTolFactor;
  float ProbRelTolFactor;
  bool RemainFeasible;
} g_struct_T;

#endif /* typedef_g_struct_T */

#ifndef typedef_h_struct_T
#define typedef_h_struct_T

typedef struct {
  bool fevalOK;
  bool done;
  bool stepAccepted;
  bool failedLineSearch;
  int stepType;
} h_struct_T;

#endif /* typedef_h_struct_T */

#ifndef typedef_i_struct_T
#define typedef_i_struct_T

typedef struct {
  float penaltyParam;
  float threshold;
  int nPenaltyDecreases;
  float linearizedConstrViol;
  float initFval;
  float initConstrViolationEq;
  float initConstrViolationIneq;
  float phi;
  float phiPrimePlus;
  float phiFullStep;
  float feasRelativeFactor;
  float nlpPrimalFeasError;
  float nlpDualFeasError;
  float nlpComplError;
  float b_firstOrderOpt;
  bool hasObjective;
} i_struct_T;

#endif /* typedef_i_struct_T */

#ifndef typedef_j_struct_T
#define typedef_j_struct_T

typedef struct {
  float grad[5];
  float Hx[4];
  bool hasLinear;
  int b_nvar;
  int maxVar;
  float beta;
  float rho;
  int objtype;
  int prev_objtype;
  int prev_nvar;
  bool prev_hasLinear;
  float gammaScalar;
} j_struct_T;

#endif /* typedef_j_struct_T */

#ifndef typedef_k_struct_T
#define typedef_k_struct_T

typedef struct {
  float FMat[81];
  int ldm;
  int ndims;
  int info;
  float scaleFactor;
  bool ConvexCheck;
  float regTol_;
  float b_workspace_;
  float workspace2_;
} k_struct_T;

#endif /* typedef_k_struct_T */

/* Function Declarations */
__device__ bool BFGSUpdate(int b_nvar, float b_Bk[16], const float sk[5],
                           float yk[5], float workspace[45]);
__device__ void PresolveWorkingSet(d_struct_T *solution, b_struct_T *memspace,
                                   e_struct_T *b_workingset,
                                   c_struct_T *b_qrmanager,
                                   const g_struct_T *options);
__device__ void RemoveDependentIneq_(e_struct_T *b_workingset,
                                     c_struct_T *b_qrmanager,
                                     b_struct_T *memspace, float tolfactor);
__device__ void addAineqConstr(e_struct_T *obj);
__device__ void addBoundToActiveSetMatrix_(e_struct_T *obj, int TYPE,
                                           int idx_local);
__device__ void b_computeGradLag(float workspace[45], int nVar,
                                 const float grad[5], int mIneq, int mEq,
                                 const int finiteFixed[5], int mFixed,
                                 const int finiteLB[5], int mLB,
                                 const int finiteUB[5], int mUB,
                                 const float lambda[9]);
__device__ void b_driver(const float lb[4], const float ub[4],
                         d_struct_T *TrialState, i_struct_T *MeritFunction,
                         const i_coder_internal_stickyStruct *FcnEvaluator,
                         f_struct_T *FiniteDifferences, b_struct_T *memspace,
                         e_struct_T *WorkingSet, float Hessian[16],
                         c_struct_T *QRManager, k_struct_T *CholManager,
                         j_struct_T *QPObjective, float optimality_tolerance);
__device__ float b_maxConstraintViolation(const e_struct_T *obj,
                                          const float x[5]);
__device__ void b_test_exit(h_struct_T *Flags, b_struct_T *memspace,
                            i_struct_T *MeritFunction, e_struct_T *WorkingSet,
                            d_struct_T *TrialState, c_struct_T *QRManager,
                            const float lb[4], const float ub[4],
                            float optimality_tolerance);
__device__ void b_xgemm(int m, int n, int k, const float b_A[81], int ia0,
                        int b_lda, const float b_B[45], float C[81], int ldc);
__device__ float b_xnrm2(int n, const float x[5]);
__device__ float computeComplError(const float xCurrent[4], int mIneq,
                                   const int finiteLB[5], int mLB,
                                   const float lb[4], const int finiteUB[5],
                                   int mUB, const float ub[4],
                                   const float lambda[9], int iL0);
__device__ bool computeFiniteDifferences(f_struct_T *obj, float fCurrent,
                                         float xk[4], float gradf[5],
                                         const float lb[4], const float ub[4]);
__device__ float computeFval(const j_struct_T *obj, float workspace[45],
                             const float b_H[16], const float f[5],
                             const float x[5]);
__device__ float computeFval_ReuseHx(const j_struct_T *obj, float workspace[45],
                                     const float f[5], const float x[5]);
__device__ void computeGradLag(float workspace[5], int nVar,
                               const float grad[5], int mIneq, int mEq,
                               const int finiteFixed[5], int mFixed,
                               const int finiteLB[5], int mLB,
                               const int finiteUB[5], int mUB,
                               const float lambda[9]);
__device__ void computeGrad_StoreHx(j_struct_T *obj, const float b_H[16],
                                    const float f[5], const float x[5]);
__device__ void computeLambdaLSQ(int nVar, int mConstr, c_struct_T *QRManager,
                                 const float ATwset[45], int ldA,
                                 const float grad[5], float lambdaLSQ[9],
                                 float workspace[45]);
__device__ float computePrimalFeasError(const float x[4], const int finiteLB[5],
                                        int mLB, const float lb[4],
                                        const int finiteUB[5], int mUB,
                                        const float ub[4]);
__device__ void computeQ_(c_struct_T *obj, int nrows);
__device__ void compute_deltax(const float b_H[16], d_struct_T *solution,
                               b_struct_T *memspace,
                               const c_struct_T *b_qrmanager,
                               k_struct_T *b_cholmanager,
                               const j_struct_T *objective,
                               bool alwaysPositiveDef);
__device__ void countsort(int x[9], int xLen, int workspace[9], int xMin,
                          int xMax);
__device__ void deleteColMoveEnd(c_struct_T *obj, int idx);
__device__ int div_nde_s32_floor(int numerator);
__device__ void driver(const float b_H[16], const float f[5],
                       d_struct_T *solution, b_struct_T *memspace,
                       e_struct_T *b_workingset, c_struct_T *b_qrmanager,
                       k_struct_T *b_cholmanager, j_struct_T *objective,
                       const g_struct_T options, g_struct_T runTimeOptions);
__device__ void factorQR(c_struct_T *obj, const float b_A[45], int mrows,
                         int ncols, int ldA);
__device__ bool feasibleX0ForWorkingSet(float workspace[45], float xCurrent[5],
                                        const e_struct_T *b_workingset,
                                        c_struct_T *b_qrmanager);
__device__ float feasibleratiotest(
    const float solution_xstar[5], const float solution_searchDir[5],
    const float workspace[45], int workingset_nVar,
    const float workingset_lb[5], const float workingset_ub[5],
    const int workingset_indexLB[5], const int workingset_indexUB[5],
    const int b_workingset_sizes[5], const int b_workingset_isActiveIdx[6],
    const bool workingset_isActiveConstr[9], const int workingset_nWConstr[5],
    bool isPhaseOne, float tolcon, bool *newBlocking, int *constrType,
    int *constrIdx);
__device__ void fullColLDL2_(k_struct_T *obj, int NColsRemain,
                             float REG_PRIMAL);
__device__ void initActiveSet(e_struct_T *obj);
__device__ void iterate(
    const float b_H[16], const float f[5], d_struct_T *solution,
    b_struct_T *memspace, e_struct_T *b_workingset, c_struct_T *b_qrmanager,
    k_struct_T *b_cholmanager, j_struct_T *objective,
    const char options_SolverName[7], float options_StepTolerance,
    float options_ConstraintTolerance, float options_ObjectiveLimit,
    float options_PricingTolerance, const g_struct_T runTimeOptions);
__device__ int ixamax(int n, const float x[81], int incx);
__device__ void linearForm_(bool obj_hasLinear, int obj_nvar,
                            float workspace[45], const float b_H[16],
                            const float f[5], const float x[5]);
__device__ float maxConstraintViolation(const e_struct_T *obj,
                                        const float x[45], int ix0);
__device__ void modifyOverheadPhaseOne_(e_struct_T *obj);
__device__ void qrf(float b_A[81], int m, int n, int nfxd, float tau[9]);
__device__ float ratiotest(
    const float solution_xstar[5], const float solution_searchDir[5],
    const float workspace[45], int workingset_nVar,
    const float workingset_lb[5], const float workingset_ub[5],
    const int workingset_indexLB[5], const int workingset_indexUB[5],
    const int b_workingset_sizes[5], const int b_workingset_isActiveIdx[6],
    const bool workingset_isActiveConstr[9], const int workingset_nWConstr[5],
    float tolcon, float *b_toldelta, bool *newBlocking, int *constrType,
    int *constrIdx);
__device__ void relaxed(const float Hessian[16], const float grad[5],
                        d_struct_T *TrialState, i_struct_T *MeritFunction,
                        b_struct_T *memspace, e_struct_T *WorkingSet,
                        c_struct_T *QRManager, k_struct_T *CholManager,
                        j_struct_T *QPObjective, g_struct_T *qpoptions);
__device__ void removeConstr(e_struct_T *obj, int idx_global);
__device__ float rt_hypotf(float u0, float u1);
__device__ void setProblemType(e_struct_T *obj, int PROBLEM_TYPE);
__device__ void solve(const k_struct_T *obj, float rhs[5]);
__device__ void sortLambdaQP(float lambda[9], int WorkingSet_nActiveConstr,
                             const int WorkingSet_sizes[5],
                             const int WorkingSet_isActiveIdx[6],
                             const int WorkingSet_Wid[9],
                             const int WorkingSet_Wlocalidx[9],
                             float workspace[45]);
__device__ float sqp_cuda_anonFcn1(const float pi4fvek_c[5],
                                   const float mmvek[10],
                                   const bool valid_freq[5], const float x[4]);
__device__ bool step(int *b_STEP_TYPE, float Hessian[16], const float lb[4],
                     const float ub[4], d_struct_T *TrialState,
                     i_struct_T *MeritFunction, b_struct_T *memspace,
                     e_struct_T *WorkingSet, c_struct_T *QRManager,
                     k_struct_T *CholManager, j_struct_T *QPObjective,
                     g_struct_T qpoptions);
__device__ bool test_exit(b_struct_T *memspace, i_struct_T *MeritFunction,
                          e_struct_T *WorkingSet, d_struct_T *TrialState,
                          c_struct_T *QRManager, const float lb[4],
                          const float ub[4], bool *Flags_fevalOK,
                          bool *Flags_done, bool *Flags_stepAccepted,
                          bool *Flags_failedLineSearch, int *Flags_stepType,
                          float optimality_tolerance);
__device__ void updateWorkingSetForNewQP(const float xk[4],
                                         e_struct_T *WorkingSet, int mEq,
                                         int mLB, const float lb[4], int mUB,
                                         const float ub[4], int mFixed);
__device__ void xgemm(int m, int n, int k, const float b_A[16], int b_lda,
                      const float b_B[81], int ib0, int ldb, float C[45]);
__device__ void xgeqp3(float b_A[81], int m, int n, int jpvt[9], float tau[9]);
__device__ float xnrm2(int n, const float x[81], int ix0);
__device__ int xpotrf(int n, float b_A[81], int b_lda);
__device__ float xrotg(float *a, float *b, float *s);
__device__ void xzlarf(int m, int n, int iv0, float tau, float C[81], int ic0,
                       float work[9]);
__device__ float xzlarfg(int n, float *alpha1, float x[81], int ix0);

/* Function Definitions */
/*
 * Arguments    : int b_nvar
 *                float b_Bk[16]
 *                const float sk[5]
 *                float yk[5]
 *                float workspace[45]
 * Return Type  : bool
 */
__device__ bool BFGSUpdate(int b_nvar, float b_Bk[16], const float sk[5],
                           float yk[5], float workspace[45]) {
  float curvatureS;
  float dotSY;
  float theta;
  int b_i;
  int i1;
  int ia;
  int iac;
  int ix;
  int k;
  bool success;
  dotSY = 0.0F;
  if (b_nvar >= 1) {
    for (k = 0; k < b_nvar; k++) {
      dotSY += sk[k] * yk[k];
    }
  }

  if (b_nvar != 0) {
    if (b_nvar - 1 >= 0) {
      (void)memset(&workspace[0], 0, (unsigned int)b_nvar * sizeof(float));
    }

    ix = 0;
    b_i = 4 * (b_nvar - 1) + 1;
    for (iac = 1; iac <= b_i; iac += 4) {
      i1 = (iac + b_nvar) - 1;
      for (ia = iac; ia <= i1; ia++) {
        k = ia - iac;
        workspace[k] += b_Bk[ia - 1] * sk[ix];
      }

      ix++;
    }
  }

  curvatureS = 0.0F;
  if (b_nvar >= 1) {
    for (k = 0; k < b_nvar; k++) {
      curvatureS += sk[k] * workspace[k];
    }
  }

  if (dotSY < 0.2F * curvatureS) {
    theta = 0.8F * curvatureS / (curvatureS - dotSY);
    for (k = 0; k < b_nvar; k++) {
      yk[k] *= theta;
    }

    if ((b_nvar >= 1) && (1.0F - theta != 0.0F)) {
      ix = b_nvar - 1;
      for (k = 0; k <= ix; k++) {
        yk[k] += (1.0F - theta) * workspace[k];
      }
    }

    dotSY = 0.0F;
    if (b_nvar >= 1) {
      for (k = 0; k < b_nvar; k++) {
        dotSY += sk[k] * yk[k];
      }
    }
  }

  if ((curvatureS > 1.1920929E-7F) && (dotSY > 1.1920929E-7F)) {
    success = true;
  } else {
    success = false;
  }

  if (success) {
    curvatureS = -1.0F / curvatureS;
    if (curvatureS != 0.0F) {
      ix = 0;
      for (k = 0; k < b_nvar; k++) {
        theta = workspace[k];
        if (theta != 0.0F) {
          theta *= curvatureS;
          b_i = ix + 1;
          i1 = b_nvar + ix;
          for (iac = b_i; iac <= i1; iac++) {
            b_Bk[iac - 1] += workspace[(iac - ix) - 1] * theta;
          }
        }

        ix += 4;
      }
    }

    curvatureS = 1.0F / dotSY;
    if (curvatureS != 0.0F) {
      ix = 0;
      for (k = 0; k < b_nvar; k++) {
        theta = yk[k];
        if (theta != 0.0F) {
          theta *= curvatureS;
          b_i = ix + 1;
          i1 = b_nvar + ix;
          for (iac = b_i; iac <= i1; iac++) {
            b_Bk[iac - 1] += yk[(iac - ix) - 1] * theta;
          }
        }

        ix += 4;
      }
    }
  }

  return success;
}

/*
 * Arguments    : d_struct_T *solution
 *                b_struct_T *memspace
 *                e_struct_T *b_workingset
 *                c_struct_T *b_qrmanager
 *                const g_struct_T *options
 * Return Type  : void
 */
__device__ void PresolveWorkingSet(d_struct_T *solution, b_struct_T *memspace,
                                   e_struct_T *b_workingset,
                                   c_struct_T *b_qrmanager,
                                   const g_struct_T *options) {
  float tol;
  int idxDiag;
  int idx_col;
  int ix;
  int k;
  int mTotalWorkingEq_tmp_tmp;
  int mWorkingFixed;
  int nDepInd;
  int nVar;
  int u0;
  solution->state = 82;
  nVar = b_workingset->nVar - 1;
  mWorkingFixed = b_workingset->nWConstr[0];
  mTotalWorkingEq_tmp_tmp =
      b_workingset->nWConstr[0] + b_workingset->nWConstr[1];
  nDepInd = 0;
  if (mTotalWorkingEq_tmp_tmp > 0) {
    int b_i;
    for (idxDiag = 0; idxDiag < mTotalWorkingEq_tmp_tmp; idxDiag++) {
      for (idx_col = 0; idx_col <= nVar; idx_col++) {
        b_qrmanager->QR[idxDiag + b_qrmanager->ldq * idx_col] =
            b_workingset->ATwset[idx_col + b_workingset->ldA * idxDiag];
      }
    }

    nDepInd = mTotalWorkingEq_tmp_tmp - b_workingset->nVar;
    if (nDepInd <= 0) {
      nDepInd = 0;
    }

    if (nVar >= 0) {
      (void)memset(&b_qrmanager->jpvt[0], 0,
                   (unsigned int)(int)(nVar + 1) * sizeof(int));
    }

    b_i = mTotalWorkingEq_tmp_tmp * b_workingset->nVar;
    if (b_i == 0) {
      b_qrmanager->mrows = mTotalWorkingEq_tmp_tmp;
      b_qrmanager->ncols = b_workingset->nVar;
      b_qrmanager->minRowCol = 0;
    } else {
      b_qrmanager->usedPivoting = true;
      b_qrmanager->mrows = mTotalWorkingEq_tmp_tmp;
      b_qrmanager->ncols = b_workingset->nVar;
      idxDiag = b_workingset->nVar;
      if (mTotalWorkingEq_tmp_tmp <= idxDiag) {
        idxDiag = mTotalWorkingEq_tmp_tmp;
      }

      b_qrmanager->minRowCol = idxDiag;
      xgeqp3(b_qrmanager->QR, mTotalWorkingEq_tmp_tmp, b_workingset->nVar,
             b_qrmanager->jpvt, b_qrmanager->tau);
    }

    tol = 100.0F * (float)b_workingset->nVar * 1.1920929E-7F;
    u0 = b_workingset->nVar;
    if (u0 > mTotalWorkingEq_tmp_tmp) {
      u0 = mTotalWorkingEq_tmp_tmp;
    }

    idxDiag = u0 + b_qrmanager->ldq * (u0 - 1);
    while ((idxDiag > 0) && (fabsf(b_qrmanager->QR[idxDiag - 1]) < tol)) {
      idxDiag = (idxDiag - b_qrmanager->ldq) - 1;
      nDepInd++;
    }

    if (nDepInd > 0) {
      bool exitg1;
      computeQ_(b_qrmanager, b_qrmanager->mrows);
      idxDiag = 0;
      exitg1 = false;
      while ((!exitg1) && (idxDiag <= nDepInd - 1)) {
        float qtb;
        ix = b_qrmanager->ldq * ((mTotalWorkingEq_tmp_tmp - idxDiag) - 1);
        qtb = 0.0F;
        for (k = 0; k < mTotalWorkingEq_tmp_tmp; k++) {
          qtb += b_qrmanager->Q[ix + k] * b_workingset->bwset[k];
        }

        if (fabsf(qtb) >= tol) {
          nDepInd = -1;
          exitg1 = true;
        } else {
          idxDiag++;
        }
      }
    }

    if (nDepInd > 0) {
      for (idx_col = 0; idx_col < mTotalWorkingEq_tmp_tmp; idx_col++) {
        idxDiag = b_qrmanager->ldq * idx_col;
        ix = b_workingset->ldA * idx_col;
        for (k = 0; k <= nVar; k++) {
          b_qrmanager->QR[idxDiag + k] = b_workingset->ATwset[ix + k];
        }
      }

      for (idxDiag = 0; idxDiag < mWorkingFixed; idxDiag++) {
        b_qrmanager->jpvt[idxDiag] = 1;
      }

      idxDiag = b_workingset->nWConstr[0] + 1;
      if (idxDiag <= mTotalWorkingEq_tmp_tmp) {
        (void)memset(
            &b_qrmanager->jpvt[idxDiag + -1], 0,
            (unsigned int)(int)((mTotalWorkingEq_tmp_tmp - idxDiag) + 1) *
                sizeof(int));
      }

      if (b_i == 0) {
        b_qrmanager->mrows = b_workingset->nVar;
        b_qrmanager->ncols = mTotalWorkingEq_tmp_tmp;
        b_qrmanager->minRowCol = 0;
      } else {
        b_qrmanager->usedPivoting = true;
        b_qrmanager->mrows = b_workingset->nVar;
        b_qrmanager->ncols = mTotalWorkingEq_tmp_tmp;
        b_qrmanager->minRowCol = u0;
        xgeqp3(b_qrmanager->QR, b_workingset->nVar, mTotalWorkingEq_tmp_tmp,
               b_qrmanager->jpvt, b_qrmanager->tau);
      }

      for (idxDiag = 0; idxDiag < nDepInd; idxDiag++) {
        memspace->workspace_int[idxDiag] =
            b_qrmanager->jpvt[(mTotalWorkingEq_tmp_tmp - nDepInd) + idxDiag];
      }

      countsort(memspace->workspace_int, nDepInd, memspace->workspace_sort, 1,
                mTotalWorkingEq_tmp_tmp);
      for (idxDiag = nDepInd; idxDiag >= 1; idxDiag--) {
        b_i = memspace->workspace_int[idxDiag - 1];
        if (b_i <= mTotalWorkingEq_tmp_tmp) {
          if ((b_workingset->nActiveConstr == mTotalWorkingEq_tmp_tmp) ||
              (b_i == mTotalWorkingEq_tmp_tmp)) {
            b_workingset->mEqRemoved++;

            /* A check that is always false is detected at compile-time.
             * Eliminating code that follows. */
          } else {
            b_workingset->mEqRemoved++;

            /* A check that is always false is detected at compile-time.
             * Eliminating code that follows. */
          }
        }
      }
    }
  }

  if ((nDepInd != -1) && (b_workingset->nActiveConstr <= b_qrmanager->ldq)) {
    bool guard1;
    bool okWorkingSet;
    RemoveDependentIneq_(b_workingset, b_qrmanager, memspace, 100.0F);
    okWorkingSet = feasibleX0ForWorkingSet(
        memspace->workspace_float, solution->xstar, b_workingset, b_qrmanager);
    guard1 = false;
    if (!okWorkingSet) {
      RemoveDependentIneq_(b_workingset, b_qrmanager, memspace, 1000.0F);
      okWorkingSet =
          feasibleX0ForWorkingSet(memspace->workspace_float, solution->xstar,
                                  b_workingset, b_qrmanager);
      if (!okWorkingSet) {
        solution->state = -7;
      } else {
        guard1 = true;
      }
    } else {
      guard1 = true;
    }

    if (guard1 && (b_workingset->nWConstr[0] + b_workingset->nWConstr[1] ==
                   b_workingset->nVar)) {
      tol = b_maxConstraintViolation(b_workingset, solution->xstar);
      if (tol > options->ConstraintTolerance) {
        solution->state = -2;
      }
    }
  } else {
    solution->state = -3;
    idxDiag = mTotalWorkingEq_tmp_tmp + 1;
    ix = b_workingset->nActiveConstr;
    for (u0 = idxDiag; u0 <= ix; u0++) {
      b_workingset->isActiveConstr
          [(b_workingset->isActiveIdx[b_workingset->Wid[u0 - 1] - 1] +
            b_workingset->Wlocalidx[u0 - 1]) -
           2] = false;
    }

    b_workingset->nWConstr[2] = 0;
    b_workingset->nWConstr[3] = 0;
    b_workingset->nWConstr[4] = 0;
    b_workingset->nActiveConstr = mTotalWorkingEq_tmp_tmp;
  }
}

/*
 * Arguments    : e_struct_T *b_workingset
 *                c_struct_T *b_qrmanager
 *                b_struct_T *memspace
 *                float tolfactor
 * Return Type  : void
 */
__device__ void RemoveDependentIneq_(e_struct_T *b_workingset,
                                     c_struct_T *b_qrmanager,
                                     b_struct_T *memspace, float tolfactor) {
  int idx;
  int k;
  int nActiveConstr_tmp;
  int nFixedConstr;
  int nVar;
  nActiveConstr_tmp = b_workingset->nActiveConstr;
  nFixedConstr = b_workingset->nWConstr[0] + b_workingset->nWConstr[1];
  nVar = b_workingset->nVar;
  if ((b_workingset->nWConstr[2] + b_workingset->nWConstr[3]) +
          b_workingset->nWConstr[4] >
      0) {
    float tol;
    int idxDiag;
    int nDepIneq;
    tol = tolfactor * (float)b_workingset->nVar * 1.1920929E-7F;
    for (idx = 0; idx < nFixedConstr; idx++) {
      b_qrmanager->jpvt[idx] = 1;
    }

    idxDiag = nFixedConstr + 1;
    if (idxDiag <= nActiveConstr_tmp) {
      (void)memset(
          &b_qrmanager->jpvt[idxDiag + -1], 0,
          (unsigned int)(int)((nActiveConstr_tmp - idxDiag) + 1) * sizeof(int));
    }

    for (idx = 0; idx < nActiveConstr_tmp; idx++) {
      idxDiag = b_qrmanager->ldq * idx;
      nDepIneq = b_workingset->ldA * idx;
      for (k = 0; k < nVar; k++) {
        b_qrmanager->QR[idxDiag + k] = b_workingset->ATwset[nDepIneq + k];
      }
    }

    if (b_workingset->nVar * b_workingset->nActiveConstr == 0) {
      b_qrmanager->mrows = b_workingset->nVar;
      b_qrmanager->ncols = b_workingset->nActiveConstr;
      b_qrmanager->minRowCol = 0;
    } else {
      b_qrmanager->usedPivoting = true;
      b_qrmanager->mrows = b_workingset->nVar;
      b_qrmanager->ncols = b_workingset->nActiveConstr;
      idxDiag = b_workingset->nVar;
      nDepIneq = b_workingset->nActiveConstr;
      if (idxDiag <= nDepIneq) {
        nDepIneq = idxDiag;
      }

      b_qrmanager->minRowCol = nDepIneq;
      xgeqp3(b_qrmanager->QR, b_workingset->nVar, b_workingset->nActiveConstr,
             b_qrmanager->jpvt, b_qrmanager->tau);
    }

    nDepIneq = 0;
    idx = b_workingset->nActiveConstr - 1;
    while (idx + 1 > nVar) {
      nDepIneq++;
      memspace->workspace_int[nDepIneq - 1] = b_qrmanager->jpvt[idx];
      idx--;
    }

    if (idx + 1 <= b_workingset->nVar) {
      idxDiag = idx + b_qrmanager->ldq * idx;
      while ((idx + 1 > nFixedConstr) &&
             (fabsf(b_qrmanager->QR[idxDiag]) < tol)) {
        nDepIneq++;
        memspace->workspace_int[nDepIneq - 1] = b_qrmanager->jpvt[idx];
        idx--;
        idxDiag = (idxDiag - b_qrmanager->ldq) - 1;
      }
    }

    countsort(memspace->workspace_int, nDepIneq, memspace->workspace_sort,
              nFixedConstr + 1, b_workingset->nActiveConstr);
    for (idx = nDepIneq; idx >= 1; idx--) {
      removeConstr(b_workingset, memspace->workspace_int[idx - 1]);
    }
  }
}

/*
 * Arguments    : e_struct_T *obj
 * Return Type  : void
 */
__device__ void addAineqConstr(e_struct_T *obj) {
  obj->nActiveConstr++;

  /* A check that is always false is detected at compile-time. Eliminating code
   * that follows. */
}

/*
 * Arguments    : e_struct_T *obj
 *                int TYPE
 *                int idx_local
 * Return Type  : void
 */
__device__ void addBoundToActiveSetMatrix_(e_struct_T *obj, int TYPE,
                                           int idx_local) {
  int b_i;
  int colOffset;
  int idx_bnd_local;
  obj->nWConstr[TYPE - 1]++;
  obj->isActiveConstr[(obj->isActiveIdx[TYPE - 1] + idx_local) - 2] = true;
  obj->nActiveConstr++;
  b_i = obj->nActiveConstr - 1;
  obj->Wid[b_i] = TYPE;
  obj->Wlocalidx[b_i] = idx_local;
  colOffset = obj->ldA * b_i - 1;
  if (TYPE == 5) {
    idx_bnd_local = obj->indexUB[idx_local - 1];
    obj->bwset[b_i] = obj->ub[idx_bnd_local - 1];
  } else {
    idx_bnd_local = obj->indexLB[idx_local - 1];
    obj->bwset[b_i] = obj->lb[idx_bnd_local - 1];
  }

  if (idx_bnd_local - 2 >= 0) {
    (void)memset(
        &obj->ATwset[colOffset + 1], 0,
        (unsigned int)(int)(((idx_bnd_local + colOffset) - colOffset) - 1) *
            sizeof(float));
  }

  obj->ATwset[idx_bnd_local + colOffset] =
      2.0F * (float)(TYPE == 5 ? (int)1 : (int)0) - 1.0F;
  b_i = idx_bnd_local + 1;
  idx_bnd_local = obj->nVar;
  if (b_i <= idx_bnd_local) {
    (void)memset(
        &obj->ATwset[b_i + colOffset], 0,
        (unsigned int)(int)((((idx_bnd_local + colOffset) - b_i) - colOffset) +
                            1) *
            sizeof(float));
  }

  switch (obj->probType) {
    case 3:
    case 2:
      break;

    default:
      obj->ATwset[obj->nVar + colOffset] = -1.0F;
      break;
  }
}

/*
 * Arguments    : float workspace[45]
 *                int nVar
 *                const float grad[5]
 *                int mIneq
 *                int mEq
 *                const int finiteFixed[5]
 *                int mFixed
 *                const int finiteLB[5]
 *                int mLB
 *                const int finiteUB[5]
 *                int mUB
 *                const float lambda[9]
 * Return Type  : void
 */
__device__ void b_computeGradLag(float workspace[45], int nVar,
                                 const float grad[5], int mIneq, int mEq,
                                 const int finiteFixed[5], int mFixed,
                                 const int finiteLB[5], int mLB,
                                 const int finiteUB[5], int mUB,
                                 const float lambda[9]) {
  int b_i;
  int iL0;
  int idx;
  if (nVar - 1 >= 0) {
    (void)memcpy(&workspace[0], &grad[0], (unsigned int)nVar * sizeof(float));
  }

  for (idx = 0; idx < mFixed; idx++) {
    b_i = finiteFixed[idx];
    workspace[b_i - 1] += lambda[idx];
  }

  iL0 = (mFixed + mEq) + mIneq;
  for (idx = 0; idx < mLB; idx++) {
    b_i = finiteLB[idx];
    workspace[b_i - 1] -= lambda[iL0 + idx];
  }

  if (mLB - 1 >= 0) {
    iL0 += mLB;
  }

  for (idx = 0; idx < mUB; idx++) {
    b_i = finiteUB[idx];
    workspace[b_i - 1] += lambda[iL0 + idx];
  }
}

/*
 * Arguments    : const float lb[4]
 *                const float ub[4]
 *                d_struct_T *TrialState
 *                i_struct_T *MeritFunction
 *                const i_coder_internal_stickyStruct *FcnEvaluator
 *                f_struct_T *FiniteDifferences
 *                b_struct_T *memspace
 *                e_struct_T *WorkingSet
 *                float Hessian[16]
 *                c_struct_T *QRManager
 *                k_struct_T *CholManager
 *                j_struct_T *QPObjective
 * Return Type  : void
 */
__device__ void b_driver(const float lb[4], const float ub[4],
                         d_struct_T *TrialState, i_struct_T *MeritFunction,
                         const i_coder_internal_stickyStruct *FcnEvaluator,
                         f_struct_T *FiniteDifferences, b_struct_T *memspace,
                         e_struct_T *WorkingSet, float Hessian[16],
                         c_struct_T *QRManager, k_struct_T *CholManager,
                         j_struct_T *QPObjective, float optimality_tolerance) {
  static const signed char iv[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1};

  static const char qpoptions_SolverName[7] = {'f', 'm', 'i', 'n',
                                               'c', 'o', 'n'};

  g_struct_T expl_temp;
  h_struct_T Flags;
  int b_i;
  int c_i;
  int ia;
  int mConstr;
  int mEq;
  int mFixed;
  int mLB;
  int mUB;
  int nVar_tmp_tmp;
  int qpoptions_MaxIterations;
  int u1;
  for (b_i = 0; b_i < 5; b_i++) {
    QPObjective->grad[b_i] = 0.0F;
  }

  QPObjective->Hx[0] = 0.0F;
  QPObjective->Hx[1] = 0.0F;
  QPObjective->Hx[2] = 0.0F;
  QPObjective->Hx[3] = 0.0F;
  QPObjective->hasLinear = true;
  QPObjective->b_nvar = 4;
  QPObjective->maxVar = 5;
  QPObjective->beta = 0.0F;
  QPObjective->rho = 0.0F;
  QPObjective->objtype = 3;
  QPObjective->prev_objtype = 3;
  QPObjective->prev_nvar = 0;
  QPObjective->prev_hasLinear = false;
  QPObjective->gammaScalar = 0.0F;
  (void)memset(&CholManager->FMat[0], 0, 81U * sizeof(float));
  CholManager->ldm = 9;
  CholManager->ndims = 0;
  CholManager->info = 0;
  CholManager->scaleFactor = 0.0F;
  CholManager->ConvexCheck = true;
  CholManager->regTol_ = 3.402823466E+38F;
  CholManager->b_workspace_ = 3.402823466E+38F;
  CholManager->workspace2_ = 3.402823466E+38F;
  for (c_i = 0; c_i < 16; c_i++) {
    Hessian[c_i] = (float)iv[c_i];
  }

  nVar_tmp_tmp = WorkingSet->nVar - 1;
  mFixed = WorkingSet->sizes[0] - 1;
  mEq = WorkingSet->sizes[1];
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mConstr =
      (((WorkingSet->sizes[0] + WorkingSet->sizes[1]) + WorkingSet->sizes[2]) +
       WorkingSet->sizes[3]) +
      WorkingSet->sizes[4];
  b_i = WorkingSet->nVar;
  u1 = ((WorkingSet->sizes[2] + WorkingSet->sizes[3]) + WorkingSet->sizes[4]) +
       2 * WorkingSet->sizes[0];
  if (b_i >= u1) {
    u1 = b_i;
  }

  qpoptions_MaxIterations = 10 * u1;
  TrialState->steplength = 1.0F;
  QRManager->ldq = 9;
  (void)memset(&QRManager->QR[0], 0, 81U * sizeof(float));
  (void)memset(&QRManager->Q[0], 0, 81U * sizeof(float));
  QRManager->mrows = 0;
  QRManager->ncols = 0;
  for (b_i = 0; b_i < 9; b_i++) {
    QRManager->jpvt[b_i] = 0;
    QRManager->tau[b_i] = 0.0F;
  }

  QRManager->minRowCol = 0;
  QRManager->usedPivoting = false;
  (void)test_exit(memspace, MeritFunction, WorkingSet, TrialState, QRManager,
                  lb, ub, &Flags.fevalOK, &Flags.done, &Flags.stepAccepted,
                  &Flags.failedLineSearch, &Flags.stepType,
                  optimality_tolerance);
  TrialState->sqpFval_old = TrialState->sqpFval;
  TrialState->xstarsqp_old[0] = TrialState->xstarsqp[0];
  TrialState->grad_old[0] = TrialState->grad[0];
  TrialState->xstarsqp_old[1] = TrialState->xstarsqp[1];
  TrialState->grad_old[1] = TrialState->grad[1];
  TrialState->xstarsqp_old[2] = TrialState->xstarsqp[2];
  TrialState->grad_old[2] = TrialState->grad[2];
  TrialState->xstarsqp_old[3] = TrialState->xstarsqp[3];
  TrialState->grad_old[3] = TrialState->grad[3];
  if (!Flags.done) {
    TrialState->sqpIterations++;
  }

  while (!Flags.done) {
    float phi_alpha;
    while (!(Flags.stepAccepted || Flags.failedLineSearch)) {
      if (Flags.stepType != 3) {
        updateWorkingSetForNewQP(TrialState->xstarsqp, WorkingSet, mEq, mLB, lb,
                                 mUB, ub, mFixed + 1);
      }

      expl_temp.RemainFeasible = false;
      expl_temp.ProbRelTolFactor = 1.0F;
      expl_temp.ConstrRelTolFactor = 1.0F;
      expl_temp.PricingTolerance = 0.0F;
      expl_temp.ObjectiveLimit = -3.402823466E+38F;
      expl_temp.ConstraintTolerance = 0.001F;
      expl_temp.StepTolerance = 1.0E-6F;
      expl_temp.MaxIterations = qpoptions_MaxIterations;
      for (c_i = 0; c_i < 7; c_i++) {
        expl_temp.SolverName[c_i] = qpoptions_SolverName[c_i];
      }

      bool d;
      d = step(&Flags.stepType, Hessian, lb, ub, TrialState, MeritFunction,
               memspace, WorkingSet, QRManager, CholManager, QPObjective,
               expl_temp);
      Flags.stepAccepted = d;
      if (Flags.stepAccepted) {
        for (b_i = 0; b_i <= nVar_tmp_tmp; b_i++) {
          TrialState->xstarsqp[b_i] += TrialState->delta_x[b_i];
        }

        TrialState->sqpFval =
            sqp_cuda_anonFcn1(FcnEvaluator->next.next.next.next.next.next.next
                                  .next.value.workspace.pi4fvek_c,
                              FcnEvaluator->next.next.next.next.next.next.next
                                  .next.value.workspace.mmvek,
                              FcnEvaluator->next.next.next.next.next.next.next
                                  .next.value.workspace.valid_freq,
                              TrialState->xstarsqp);
        Flags.fevalOK = true;
        TrialState->FunctionEvaluations++;
        MeritFunction->phiFullStep = TrialState->sqpFval;
      }

      if ((Flags.stepType == 1) && Flags.stepAccepted && Flags.fevalOK &&
          (MeritFunction->phi < MeritFunction->phiFullStep) &&
          (TrialState->sqpFval < TrialState->sqpFval_old)) {
        Flags.stepType = 3;
        Flags.stepAccepted = false;
      } else {
        float alpha;
        bool b;
        bool socTaken;
        if ((Flags.stepType == 3) && Flags.stepAccepted) {
          socTaken = true;
        } else {
          socTaken = false;
        }

        b = Flags.fevalOK;
        c_i = WorkingSet->nVar - 1;
        alpha = 1.0F;
        b_i = 1;
        phi_alpha = MeritFunction->phiFullStep;
        if (c_i >= 0) {
          (void)memcpy(&TrialState->searchDir[0], &TrialState->delta_x[0],
                       (unsigned int)(int)(c_i + 1) * sizeof(float));
        }

        int exitg1;
        do {
          exitg1 = 0;
          if (TrialState->FunctionEvaluations < 400) {
            if (b && (phi_alpha <=
                      MeritFunction->phi +
                          alpha * 0.0001F * MeritFunction->phiPrimePlus)) {
              exitg1 = 1;
            } else {
              bool exitg2;
              bool tooSmallX;
              alpha *= 0.7F;
              for (u1 = 0; u1 <= c_i; u1++) {
                TrialState->delta_x[u1] = alpha * TrialState->xstar[u1];
              }

              if (socTaken) {
                phi_alpha = alpha * alpha;
                if ((c_i + 1 >= 1) && (phi_alpha != 0.0F)) {
                  for (u1 = 0; u1 <= c_i; u1++) {
                    TrialState->delta_x[u1] +=
                        phi_alpha * TrialState->socDirection[u1];
                  }
                }
              }

              tooSmallX = true;
              u1 = 0;
              exitg2 = false;
              while ((!exitg2) && (u1 <= c_i)) {
                if (0.0001F * fmaxf(1.0F, fabsf(TrialState->xstarsqp[u1])) <=
                    fabsf(TrialState->delta_x[u1])) {
                  tooSmallX = false;
                  exitg2 = true;
                } else {
                  u1++;
                }
              }

              if (tooSmallX) {
                b_i = -2;
                exitg1 = 1;
              } else {
                for (u1 = 0; u1 <= c_i; u1++) {
                  TrialState->xstarsqp[u1] =
                      TrialState->xstarsqp_old[u1] + TrialState->delta_x[u1];
                }

                TrialState->sqpFval = sqp_cuda_anonFcn1(
                    FcnEvaluator->next.next.next.next.next.next.next.next.value
                        .workspace.pi4fvek_c,
                    FcnEvaluator->next.next.next.next.next.next.next.next.value
                        .workspace.mmvek,
                    FcnEvaluator->next.next.next.next.next.next.next.next.value
                        .workspace.valid_freq,
                    TrialState->xstarsqp);
                TrialState->FunctionEvaluations++;
                b = true;
                phi_alpha = TrialState->sqpFval;
              }
            }
          } else {
            b_i = 0;
            exitg1 = 1;
          }
        } while (exitg1 == 0);

        Flags.fevalOK = b;
        TrialState->steplength = alpha;
        if (b_i > 0) {
          Flags.stepAccepted = true;
        } else {
          Flags.failedLineSearch = true;
        }
      }
    }

    if (Flags.stepAccepted && (!Flags.failedLineSearch)) {
      for (u1 = 0; u1 <= nVar_tmp_tmp; u1++) {
        TrialState->xstarsqp[u1] =
            TrialState->xstarsqp_old[u1] + TrialState->delta_x[u1];
      }

      for (u1 = 0; u1 < mConstr; u1++) {
        phi_alpha = TrialState->lambdasqp[u1];
        phi_alpha +=
            TrialState->steplength * (TrialState->lambda[u1] - phi_alpha);
        TrialState->lambdasqp[u1] = phi_alpha;
      }

      TrialState->sqpFval_old = TrialState->sqpFval;
      TrialState->xstarsqp_old[0] = TrialState->xstarsqp[0];
      TrialState->grad_old[0] = TrialState->grad[0];
      TrialState->xstarsqp_old[1] = TrialState->xstarsqp[1];
      TrialState->grad_old[1] = TrialState->grad[1];
      TrialState->xstarsqp_old[2] = TrialState->xstarsqp[2];
      TrialState->grad_old[2] = TrialState->grad[2];
      TrialState->xstarsqp_old[3] = TrialState->xstarsqp[3];
      TrialState->grad_old[3] = TrialState->grad[3];
      (void)computeFiniteDifferences(FiniteDifferences, TrialState->sqpFval,
                                     TrialState->xstarsqp, TrialState->grad, lb,
                                     ub);
      TrialState->FunctionEvaluations += FiniteDifferences->numEvals;
    } else {
      TrialState->sqpFval = TrialState->sqpFval_old;
      TrialState->xstarsqp[0] = TrialState->xstarsqp_old[0];
      TrialState->xstarsqp[1] = TrialState->xstarsqp_old[1];
      TrialState->xstarsqp[2] = TrialState->xstarsqp_old[2];
      TrialState->xstarsqp[3] = TrialState->xstarsqp_old[3];
    }

    b_test_exit(&Flags, memspace, MeritFunction, WorkingSet, TrialState,
                QRManager, lb, ub, optimality_tolerance);
    if ((!Flags.done) && Flags.stepAccepted) {
      int i1;
      int i2;
      int iac;
      Flags.stepAccepted = false;
      Flags.stepType = 1;
      Flags.failedLineSearch = false;
      if (nVar_tmp_tmp >= 0) {
        (void)memcpy(&TrialState->delta_gradLag[0], &TrialState->grad[0],
                     (unsigned int)(int)(nVar_tmp_tmp + 1) * sizeof(float));
      }

      if (nVar_tmp_tmp + 1 >= 1) {
        for (u1 = 0; u1 <= nVar_tmp_tmp; u1++) {
          TrialState->delta_gradLag[u1] -= TrialState->grad_old[u1];
        }
      }

      if (TrialState->mNonlinEq > 0) {
        u1 = WorkingSet->ldA;
        if ((nVar_tmp_tmp + 1 != 0) && (TrialState->mNonlinEq != 0)) {
          b_i = mFixed + TrialState->iNonEq0;
          c_i = WorkingSet->ldA * (TrialState->mNonlinEq - 1) + 1;
          iac = 1;
          while (((u1 > 0) && (iac <= c_i)) || ((u1 < 0) && (iac >= c_i))) {
            i1 = iac + nVar_tmp_tmp;
            for (ia = iac; ia <= i1; ia++) {
              i2 = ia - iac;
              TrialState->delta_gradLag[i2] +=
                  TrialState->delta_gradLag[ia - 1] *
                  -TrialState->lambdasqp[b_i];
            }

            b_i++;
            iac += u1;
          }
        }
      }

      if (TrialState->mNonlinIneq > 0) {
        u1 = WorkingSet->ldA;
        if ((nVar_tmp_tmp + 1 != 0) && (TrialState->mNonlinIneq != 0)) {
          b_i = (mFixed + mEq) + TrialState->iNonIneq0;
          c_i = WorkingSet->ldA * (TrialState->mNonlinIneq - 1) + 1;
          iac = 1;
          while (((u1 > 0) && (iac <= c_i)) || ((u1 < 0) && (iac >= c_i))) {
            i1 = iac + nVar_tmp_tmp;
            for (ia = iac; ia <= i1; ia++) {
              i2 = ia - iac;
              TrialState->delta_gradLag[i2] +=
                  TrialState->delta_gradLag[ia - 1] *
                  -TrialState->lambdasqp[b_i];
            }

            b_i++;
            iac += u1;
          }
        }
      }

      (void)BFGSUpdate(nVar_tmp_tmp + 1, Hessian, TrialState->delta_x,
                       TrialState->delta_gradLag, memspace->workspace_float);
      TrialState->sqpIterations++;
    }
  }
}

/*
 * Arguments    : const e_struct_T *obj
 *                const float x[5]
 * Return Type  : float
 */
__device__ float b_maxConstraintViolation(const e_struct_T *obj,
                                          const float x[5]) {
  float v;
  int idx;
  int mFixed;
  int mLB;
  int mUB;
  mLB = obj->sizes[3];
  mUB = obj->sizes[4];
  mFixed = obj->sizes[0];
  v = 0.0F;
  if (obj->sizes[3] > 0) {
    for (idx = 0; idx < mLB; idx++) {
      int idxLB;
      idxLB = obj->indexLB[idx] - 1;
      v = fmaxf(v, -x[idxLB] - obj->lb[idxLB]);
    }
  }

  if (obj->sizes[4] > 0) {
    for (idx = 0; idx < mUB; idx++) {
      mLB = obj->indexUB[idx] - 1;
      v = fmaxf(v, x[mLB] - obj->ub[mLB]);
    }
  }

  if (obj->sizes[0] > 0) {
    for (idx = 0; idx < mFixed; idx++) {
      v = fmaxf(v, fabsf(x[obj->indexFixed[idx] - 1] -
                         obj->ub[obj->indexFixed[idx] - 1]));
    }
  }

  return v;
}

/*
 * Arguments    : h_struct_T *Flags
 *                b_struct_T *memspace
 *                i_struct_T *MeritFunction
 *                e_struct_T *WorkingSet
 *                d_struct_T *TrialState
 *                c_struct_T *QRManager
 *                const float lb[4]
 *                const float ub[4]
 * Return Type  : void
 */
__device__ void b_test_exit(h_struct_T *Flags, b_struct_T *memspace,
                            i_struct_T *MeritFunction, e_struct_T *WorkingSet,
                            d_struct_T *TrialState, c_struct_T *QRManager,
                            const float lb[4], const float ub[4],
                            float optimality_tolerance) {
  float f;
  float optimRelativeFactor;
  float s;
  float smax;
  int b_idx_max;
  int k;
  int mIneq;
  int mLB;
  int mLambda;
  int mLambda_tmp;
  int mUB;
  int nVar;
  bool dxTooSmall;
  bool exitg1;
  bool isFeasible;
  nVar = WorkingSet->nVar;
  mIneq = WorkingSet->sizes[2];
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mLambda_tmp = WorkingSet->sizes[0] + WorkingSet->sizes[1];
  mLambda = (((mLambda_tmp + WorkingSet->sizes[2]) + WorkingSet->sizes[3]) +
             WorkingSet->sizes[4]) -
            1;
  if (mLambda >= 0) {
    (void)memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambdasqp[0],
                 (unsigned int)(int)(mLambda + 1) * sizeof(float));
  }

  computeGradLag(TrialState->gradLag, WorkingSet->nVar, TrialState->grad,
                 WorkingSet->sizes[2], WorkingSet->sizes[1],
                 WorkingSet->indexFixed, WorkingSet->sizes[0],
                 WorkingSet->indexLB, WorkingSet->sizes[3], WorkingSet->indexUB,
                 WorkingSet->sizes[4], TrialState->lambdaStopTest);
  if (WorkingSet->nVar < 1) {
    b_idx_max = 0;
  } else {
    b_idx_max = 1;
    if (WorkingSet->nVar > 1) {
      smax = fabsf(TrialState->grad[0]);
      for (k = 2; k <= nVar; k++) {
        s = fabsf(TrialState->grad[k - 1]);
        if (s > smax) {
          b_idx_max = k;
          smax = s;
        }
      }
    }
  }

  optimRelativeFactor = fmaxf(1.0F, fabsf(TrialState->grad[b_idx_max - 1]));
  if (optimRelativeFactor >= 3.402823466E+38F) {
    optimRelativeFactor = 1.0F;
  }

  MeritFunction->nlpPrimalFeasError = computePrimalFeasError(
      TrialState->xstarsqp, WorkingSet->indexLB, WorkingSet->sizes[3], lb,
      WorkingSet->indexUB, WorkingSet->sizes[4], ub);
  if (TrialState->sqpIterations == 0) {
    MeritFunction->feasRelativeFactor =
        fmaxf(1.0F, MeritFunction->nlpPrimalFeasError);
  }

  isFeasible = (MeritFunction->nlpPrimalFeasError <=
                0.001F * MeritFunction->feasRelativeFactor);
  dxTooSmall = true;
  s = 0.0F;
  b_idx_max = 0;
  exitg1 = false;
  while ((!exitg1) && (b_idx_max <= nVar - 1)) {
    f = fabsf(TrialState->gradLag[b_idx_max]);
    dxTooSmall = (f < 3.402823466E+38F);
    if (!dxTooSmall) {
      exitg1 = true;
    } else {
      s = fmaxf(s, f);
      b_idx_max++;
    }
  }

  MeritFunction->nlpDualFeasError = s;
  if (!dxTooSmall) {
    Flags->done = true;
    if (isFeasible) {
      TrialState->sqpExitFlag = 2;
    } else {
      TrialState->sqpExitFlag = -2;
    }
  } else {
    MeritFunction->nlpComplError = computeComplError(
        TrialState->xstarsqp, WorkingSet->sizes[2], WorkingSet->indexLB,
        WorkingSet->sizes[3], lb, WorkingSet->indexUB, WorkingSet->sizes[4], ub,
        TrialState->lambdaStopTest, mLambda_tmp + 1);
    MeritFunction->b_firstOrderOpt = fmaxf(s, MeritFunction->nlpComplError);
    if (TrialState->sqpIterations > 1) {
      float nlpDualFeasErrorTmp;
      b_computeGradLag(memspace->workspace_float, WorkingSet->nVar,
                       TrialState->grad, WorkingSet->sizes[2],
                       WorkingSet->sizes[1], WorkingSet->indexFixed,
                       WorkingSet->sizes[0], WorkingSet->indexLB,
                       WorkingSet->sizes[3], WorkingSet->indexUB,
                       WorkingSet->sizes[4], TrialState->lambdaStopTestPrev);
      nlpDualFeasErrorTmp = 0.0F;
      b_idx_max = 0;
      exitg1 = false;
      while ((!exitg1) && (b_idx_max <= nVar - 1)) {
        f = fabsf(memspace->workspace_float[b_idx_max]);
        if (f >= 3.402823466E+38F) {
          exitg1 = true;
        } else {
          nlpDualFeasErrorTmp = fmaxf(nlpDualFeasErrorTmp, f);
          b_idx_max++;
        }
      }

      smax = computeComplError(TrialState->xstarsqp, WorkingSet->sizes[2],
                               WorkingSet->indexLB, WorkingSet->sizes[3], lb,
                               WorkingSet->indexUB, WorkingSet->sizes[4], ub,
                               TrialState->lambdaStopTestPrev, mLambda_tmp + 1);
      if ((nlpDualFeasErrorTmp < s) && (smax < MeritFunction->nlpComplError)) {
        MeritFunction->nlpDualFeasError = nlpDualFeasErrorTmp;
        MeritFunction->nlpComplError = smax;
        MeritFunction->b_firstOrderOpt = fmaxf(nlpDualFeasErrorTmp, smax);
        if (mLambda >= 0) {
          (void)memcpy(&TrialState->lambdaStopTest[0],
                       &TrialState->lambdaStopTestPrev[0],
                       (unsigned int)(int)(mLambda + 1) * sizeof(float));
        }
      } else if (mLambda >= 0) {
        (void)memcpy(&TrialState->lambdaStopTestPrev[0],
                     &TrialState->lambdaStopTest[0],
                     (unsigned int)(int)(mLambda + 1) * sizeof(float));
      } else {
        /* no actions */
      }
    } else if (mLambda >= 0) {
      (void)memcpy(&TrialState->lambdaStopTestPrev[0],
                   &TrialState->lambdaStopTest[0],
                   (unsigned int)(int)(mLambda + 1) * sizeof(float));
    } else {
      /* no actions */
    }

    if (isFeasible &&
        (MeritFunction->nlpDualFeasError <=
         optimality_tolerance * optimRelativeFactor) &&
        (MeritFunction->nlpComplError <=
         optimality_tolerance * optimRelativeFactor)) {
      Flags->done = true;
      TrialState->sqpExitFlag = 1;
    } else {
      Flags->done = false;
      if (isFeasible && ((double)TrialState->sqpFval < -1.0E+20)) {
        Flags->done = true;
        TrialState->sqpExitFlag = -3;
      } else {
        bool guard1;
        guard1 = false;
        if (TrialState->sqpIterations > 0) {
          dxTooSmall = true;
          b_idx_max = 0;
          exitg1 = false;
          while ((!exitg1) && (b_idx_max <= nVar - 1)) {
            if (0.0001F * fmaxf(1.0F, fabsf(TrialState->xstarsqp[b_idx_max])) <=
                fabsf(TrialState->delta_x[b_idx_max])) {
              dxTooSmall = false;
              exitg1 = true;
            } else {
              b_idx_max++;
            }
          }

          if (dxTooSmall) {
            if (!isFeasible) {
              if (Flags->stepType != 2) {
                Flags->stepType = 2;
                Flags->failedLineSearch = false;
                Flags->stepAccepted = false;
                guard1 = true;
              } else {
                Flags->done = true;
                TrialState->sqpExitFlag = -2;
              }
            } else if (WorkingSet->nActiveConstr == 0) {
              Flags->done = true;
              TrialState->sqpExitFlag = 2;
            } else {
              if (TrialState->mNonlinEq + TrialState->mNonlinIneq > 0) {
                updateWorkingSetForNewQP(
                    TrialState->xstarsqp, WorkingSet, WorkingSet->sizes[1],
                    WorkingSet->sizes[3], lb, WorkingSet->sizes[4], ub,
                    WorkingSet->sizes[0]);
              }

              computeLambdaLSQ(nVar, WorkingSet->nActiveConstr, QRManager,
                               WorkingSet->ATwset, WorkingSet->ldA,
                               TrialState->grad, TrialState->lambda,
                               memspace->workspace_float);
              sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                           WorkingSet->sizes, WorkingSet->isActiveIdx,
                           WorkingSet->Wid, WorkingSet->Wlocalidx,
                           memspace->workspace_float);
              b_computeGradLag(memspace->workspace_float, nVar,
                               TrialState->grad, mIneq, WorkingSet->sizes[1],
                               WorkingSet->indexFixed, WorkingSet->sizes[0],
                               WorkingSet->indexLB, mLB, WorkingSet->indexUB,
                               mUB, TrialState->lambda);
              smax = 0.0F;
              b_idx_max = 0;
              exitg1 = false;
              while ((!exitg1) && (b_idx_max <= nVar - 1)) {
                f = fabsf(memspace->workspace_float[b_idx_max]);
                if (f >= 3.402823466E+38F) {
                  exitg1 = true;
                } else {
                  smax = fmaxf(smax, f);
                  b_idx_max++;
                }
              }

              s = computeComplError(TrialState->xstarsqp, mIneq,
                                    WorkingSet->indexLB, mLB, lb,
                                    WorkingSet->indexUB, mUB, ub,
                                    TrialState->lambda, mLambda_tmp + 1);
              f = fmaxf(smax, s);
              if (f <= fmaxf(MeritFunction->nlpDualFeasError,
                             MeritFunction->nlpComplError)) {
                MeritFunction->nlpDualFeasError = smax;
                MeritFunction->nlpComplError = s;
                MeritFunction->b_firstOrderOpt = f;
                if (mLambda >= 0) {
                  (void)memcpy(
                      &TrialState->lambdaStopTest[0], &TrialState->lambda[0],
                      (unsigned int)(int)(mLambda + 1) * sizeof(float));
                }
              }

              if ((MeritFunction->nlpDualFeasError <=
                   optimality_tolerance * optimRelativeFactor) &&
                  (MeritFunction->nlpComplError <=
                   optimality_tolerance * optimRelativeFactor)) {
                TrialState->sqpExitFlag = 1;
              } else {
                TrialState->sqpExitFlag = 2;
              }

              Flags->done = true;
              guard1 = true;
            }
          } else {
            guard1 = true;
          }
        } else {
          guard1 = true;
        }

        if (guard1) {
          if (TrialState->sqpIterations >= kMaxIter) {
            Flags->done = true;
            TrialState->sqpExitFlag = 0;
          } else if (TrialState->FunctionEvaluations >= 4 * kMaxIter) {
            Flags->done = true;
            TrialState->sqpExitFlag = 0;
          } else {
            /* no actions */
          }
        }
      }
    }
  }
}

/*
 * Arguments    : int m
 *                int n
 *                int k
 *                const float b_A[81]
 *                int ia0
 *                int b_lda
 *                const float b_B[45]
 *                float C[81]
 *                int ldc
 * Return Type  : void
 */
__device__ void b_xgemm(int m, int n, int k, const float b_A[81], int ia0,
                        int b_lda, const float b_B[45], float C[81], int ldc) {
  int ic;
  int w;
  if ((m != 0) && (n != 0)) {
    int b_i;
    int br;
    int cr;
    int i1;
    int lastColC;
    lastColC = ldc * (n - 1);
    cr = 0;
    while (((ldc > 0) && (cr <= lastColC)) || ((ldc < 0) && (cr >= lastColC))) {
      b_i = cr + 1;
      i1 = cr + m;
      if (b_i <= i1) {
        (void)memset(&C[b_i + -1], 0,
                     (unsigned int)(int)((i1 - b_i) + 1) * sizeof(float));
      }

      cr += ldc;
    }

    br = -1;
    cr = 0;
    while (((ldc > 0) && (cr <= lastColC)) || ((ldc < 0) && (cr >= lastColC))) {
      int ar;
      ar = ia0;
      b_i = cr + 1;
      i1 = cr + m;
      for (ic = b_i; ic <= i1; ic++) {
        float temp;
        temp = 0.0F;
        for (w = 0; w < k; w++) {
          temp += b_A[(w + ar) - 1] * b_B[(w + br) + 1];
        }

        C[ic - 1] += temp;
        ar += b_lda;
      }

      br += 9;
      cr += ldc;
    }
  }
}

/*
 * Arguments    : int n
 *                const float x[5]
 * Return Type  : float
 */
__device__ float b_xnrm2(int n, const float x[5]) {
  float y;
  int k;
  y = 0.0F;
  if (n >= 1) {
    if (n == 1) {
      y = fabsf(x[0]);
    } else {
      float scale;
      scale = 1.29246971E-26F;
      for (k = 0; k < n; k++) {
        float absxk;
        absxk = fabsf(x[k]);
        if (absxk > scale) {
          float t;
          t = scale / absxk;
          y = y * t * t + 1.0F;
          scale = absxk;
        } else {
          float t;
          t = absxk / scale;
          y += t * t;
        }
      }

      y = scale * sqrtf(y);
    }
  }

  return y;
}

/*
 * Arguments    : const float xCurrent[4]
 *                int mIneq
 *                const int finiteLB[5]
 *                int mLB
 *                const float lb[4]
 *                const int finiteUB[5]
 *                int mUB
 *                const float ub[4]
 *                const float lambda[9]
 *                int iL0
 * Return Type  : float
 */
__device__ float computeComplError(const float xCurrent[4], int mIneq,
                                   const int finiteLB[5], int mLB,
                                   const float lb[4], const int finiteUB[5],
                                   int mUB, const float ub[4],
                                   const float lambda[9], int iL0) {
  float nlpComplError;
  int idx;
  nlpComplError = 0.0F;
  if ((mIneq + mLB) + mUB > 0) {
    float lbDelta;
    float lbLambda;
    int b_i;
    int lbOffset;
    int ubOffset;
    lbOffset = (iL0 + mIneq) - 1;
    ubOffset = lbOffset + mLB;
    for (idx = 0; idx < mLB; idx++) {
      b_i = finiteLB[idx];
      lbDelta = xCurrent[b_i - 1] - lb[b_i - 1];
      lbLambda = lambda[lbOffset + idx];
      nlpComplError = fmaxf(
          nlpComplError,
          fminf(fabsf(lbDelta * lbLambda), fminf(fabsf(lbDelta), lbLambda)));
    }

    for (idx = 0; idx < mUB; idx++) {
      b_i = finiteUB[idx];
      lbDelta = ub[b_i - 1] - xCurrent[b_i - 1];
      lbLambda = lambda[ubOffset + idx];
      nlpComplError = fmaxf(
          nlpComplError,
          fminf(fabsf(lbDelta * lbLambda), fminf(fabsf(lbDelta), lbLambda)));
    }
  }

  return nlpComplError;
}

/*
 * Arguments    : f_struct_T *obj
 *                float fCurrent
 *                float xk[4]
 *                float gradf[5]
 *                const float lb[4]
 *                const float ub[4]
 * Return Type  : bool
 */
__device__ bool computeFiniteDifferences(f_struct_T *obj, float fCurrent,
                                         float xk[4], float gradf[5],
                                         const float lb[4], const float ub[4]) {
  int idx;
  bool evalOK;
  if (obj->FiniteDifferenceType == 0) {
    obj->numEvals = 0;
    for (idx = 0; idx < obj->nVar; idx++) {
      float b_deltaX;
      float distNear;
      float f;
      b_deltaX = 0.000345266977F *
                 (1.0F - 2.0F * (float)(xk[idx] < 0.0F ? (int)1 : (int)0)) *
                 fmaxf(fabsf(xk[idx]), 1.0F);
      if (obj->hasLB[idx] || obj->hasUB[idx]) {
        if (obj->hasLB[idx] && obj->hasUB[idx]) {
          float delta1;
          delta1 = b_deltaX;
          if ((lb[idx] != ub[idx]) && (xk[idx] >= lb[idx]) &&
              (xk[idx] <= ub[idx])) {
            f = xk[idx] + b_deltaX;
            if ((f > ub[idx]) || (f < lb[idx])) {
              delta1 = -b_deltaX;
              f = xk[idx] - b_deltaX;
              if ((f > ub[idx]) || (f < lb[idx])) {
                float delta2;
                delta1 = xk[idx] - lb[idx];
                delta2 = ub[idx] - xk[idx];
                if (delta1 <= delta2) {
                  delta1 = -delta1;
                } else {
                  delta1 = delta2;
                }
              }
            }
          }

          b_deltaX = delta1;
        } else if (obj->hasUB[idx]) {
          if ((xk[idx] <= ub[idx]) && (xk[idx] + b_deltaX > ub[idx])) {
            b_deltaX = -b_deltaX;
          }
        } else if ((xk[idx] >= lb[idx]) && (xk[idx] + b_deltaX < lb[idx])) {
          b_deltaX = -b_deltaX;
        } else {
          /* no actions */
        }
      }

      f = obj->f_1;
      distNear = xk[idx];
      xk[idx] += b_deltaX;
      if (!obj->SpecifyObjectiveGradient) {
        f = sqp_cuda_anonFcn1(obj->objfun.workspace.pi4fvek_c,
                              obj->objfun.workspace.mmvek,
                              obj->objfun.workspace.valid_freq, xk);
      }

      xk[idx] = distNear;
      obj->f_1 = f;
      obj->numEvals++;
      gradf[idx] = (f - fCurrent) / b_deltaX;
    }

    evalOK = true;
  } else {
    evalOK = true;
    obj->numEvals = 0;
    for (idx = 0; idx < obj->nVar; idx++) {
      float b_deltaX;
      float delta1;
      float delta2;
      float distNear;
      float f;
      int formulaType;
      b_deltaX = 0.000345266977F * fmaxf(fabsf(xk[idx]), 1.0F);
      if (obj->hasLB[idx] || obj->hasUB[idx]) {
        if (obj->hasLB[idx] && obj->hasUB[idx]) {
          formulaType = 0;
          if ((lb[idx] != ub[idx]) && (xk[idx] >= lb[idx]) &&
              (xk[idx] <= ub[idx])) {
            if (xk[idx] - b_deltaX < lb[idx]) {
              if (ub[idx] < xk[idx] + b_deltaX) {
                delta1 = xk[idx] - lb[idx];
                delta2 = ub[idx] - xk[idx];
                distNear = fminf(delta1, delta2);
                b_deltaX = fmaxf(delta1, delta2) / 2.0F;
                if (distNear >= b_deltaX) {
                  b_deltaX = distNear;
                } else if (delta1 >= delta2) {
                  formulaType = -1;
                } else {
                  formulaType = 1;
                }
              } else if (xk[idx] + 2.0F * b_deltaX <= ub[idx]) {
                formulaType = 1;
              } else {
                b_deltaX = (ub[idx] - xk[idx]) / 2.0F;
                f = xk[idx] - lb[idx];
                if (f >= b_deltaX) {
                  b_deltaX = f;
                } else {
                  formulaType = 1;
                }
              }
            } else if (ub[idx] < xk[idx] + b_deltaX) {
              if (lb[idx] <= xk[idx] - 2.0F * b_deltaX) {
                formulaType = -1;
              } else {
                b_deltaX = (xk[idx] - lb[idx]) / 2.0F;
                f = ub[idx] - xk[idx];
                if (f >= b_deltaX) {
                  b_deltaX = f;
                } else {
                  formulaType = -1;
                }
              }
            } else {
              /* no actions */
            }
          }
        } else if (obj->hasUB[idx]) {
          formulaType = 0;
          if ((xk[idx] <= ub[idx]) && (ub[idx] < xk[idx] + b_deltaX)) {
            formulaType = -1;
          }
        } else {
          formulaType = 0;
          if ((xk[idx] >= lb[idx]) && (xk[idx] - b_deltaX < lb[idx])) {
            formulaType = 1;
          }
        }
      } else {
        formulaType = 0;
      }

      switch (formulaType) {
        case 0:
          delta1 = -b_deltaX;
          delta2 = b_deltaX;
          break;

        case -1:
          delta1 = -2.0F * b_deltaX;
          delta2 = -b_deltaX;
          break;

        default:
          delta1 = b_deltaX;
          delta2 = 2.0F * b_deltaX;
          break;
      }

      f = obj->f_1;
      distNear = xk[idx];
      xk[idx] += delta1;
      if (!obj->SpecifyObjectiveGradient) {
        f = sqp_cuda_anonFcn1(obj->objfun.workspace.pi4fvek_c,
                              obj->objfun.workspace.mmvek,
                              obj->objfun.workspace.valid_freq, xk);
      }

      xk[idx] = distNear;
      obj->f_1 = f;
      obj->numEvals++;
      f = obj->f_2;
      distNear = xk[idx];
      xk[idx] += delta2;
      if (!obj->SpecifyObjectiveGradient) {
        f = sqp_cuda_anonFcn1(obj->objfun.workspace.pi4fvek_c,
                              obj->objfun.workspace.mmvek,
                              obj->objfun.workspace.valid_freq, xk);
      }

      xk[idx] = distNear;
      obj->f_2 = f;
      obj->numEvals++;
      if (!obj->SpecifyObjectiveGradient) {
        delta1 = obj->f_1;
        delta2 = obj->f_2;
        switch (formulaType) {
          case 0:
            gradf[idx] = (-delta1 + delta2) / (2.0F * b_deltaX);
            break;

          case 1:
            gradf[idx] = ((-3.0F * fCurrent + 4.0F * delta1) - delta2) /
                         (2.0F * b_deltaX);
            break;

          default:
            gradf[idx] = ((delta1 - 4.0F * delta2) + 3.0F * fCurrent) /
                         (2.0F * b_deltaX);
            break;
        }
      }
    }
  }

  return evalOK;
}

/*
 * Arguments    : const j_struct_T *obj
 *                float workspace[45]
 *                const float b_H[16]
 *                const float f[5]
 *                const float x[5]
 * Return Type  : float
 */
__device__ float computeFval(const j_struct_T *obj, float workspace[45],
                             const float b_H[16], const float f[5],
                             const float x[5]) {
  float val;
  int idx;
  int k;
  val = 0.0F;
  switch (obj->objtype) {
    case 5:
      val = obj->gammaScalar * x[obj->b_nvar - 1];
      break;

    case 3: {
      linearForm_(obj->hasLinear, obj->b_nvar, workspace, b_H, f, x);
      if (obj->b_nvar >= 1) {
        int ixlast;
        ixlast = obj->b_nvar;
        for (k = 0; k < ixlast; k++) {
          val += x[k] * workspace[k];
        }
      }
    } break;

    case 4: {
      int ixlast;
      linearForm_(obj->hasLinear, obj->b_nvar, workspace, b_H, f, x);
      ixlast = obj->b_nvar + 1;
      k = obj->maxVar - 1;
      for (idx = ixlast; idx <= k; idx++) {
        workspace[idx - 1] = 0.5F * obj->beta * x[idx - 1] + obj->rho;
      }

      if (k >= 1) {
        ixlast = obj->maxVar;
        for (k = 0; k <= ixlast - 2; k++) {
          val += x[k] * workspace[k];
        }
      }
    } break;
  }

  return val;
}

/*
 * Arguments    : const j_struct_T *obj
 *                float workspace[45]
 *                const float f[5]
 *                const float x[5]
 * Return Type  : float
 */
__device__ float computeFval_ReuseHx(const j_struct_T *obj, float workspace[45],
                                     const float f[5], const float x[5]) {
  float val;
  int k;
  val = 0.0F;
  switch (obj->objtype) {
    case 5:
      val = obj->gammaScalar * x[obj->b_nvar - 1];
      break;

    case 3: {
      if (obj->hasLinear) {
        int ixlast;
        ixlast = obj->b_nvar;
        for (k = 0; k < ixlast; k++) {
          workspace[k] = 0.5F * obj->Hx[k] + f[k];
        }

        if (obj->b_nvar >= 1) {
          ixlast = obj->b_nvar;
          for (k = 0; k < ixlast; k++) {
            val += x[k] * workspace[k];
          }
        }
      } else {
        if (obj->b_nvar >= 1) {
          int ixlast;
          ixlast = obj->b_nvar;
          for (k = 0; k < ixlast; k++) {
            val += x[k] * obj->Hx[k];
          }
        }

        val *= 0.5F;
      }
    } break;

    case 4: {
      int maxRegVar_tmp;
      maxRegVar_tmp = obj->maxVar - 1;
      if (obj->hasLinear) {
        int ixlast;
        ixlast = obj->b_nvar;
        if (ixlast - 1 >= 0) {
          (void)memcpy(&workspace[0], &f[0],
                       (unsigned int)ixlast * sizeof(float));
        }

        ixlast = obj->maxVar - obj->b_nvar;
        for (k = 0; k <= ixlast - 2; k++) {
          workspace[obj->b_nvar + k] = obj->rho;
        }

        for (k = 0; k < maxRegVar_tmp; k++) {
          workspace[k] += 0.5F * obj->Hx[k];
        }

        if (maxRegVar_tmp >= 1) {
          ixlast = obj->maxVar;
          for (k = 0; k <= ixlast - 2; k++) {
            val += x[k] * workspace[k];
          }
        }
      } else {
        int ixlast;
        if (maxRegVar_tmp >= 1) {
          ixlast = obj->maxVar;
          for (k = 0; k <= ixlast - 2; k++) {
            val += x[k] * obj->Hx[k];
          }
        }

        val *= 0.5F;
        ixlast = obj->b_nvar + 1;
        for (k = ixlast; k <= maxRegVar_tmp; k++) {
          val += x[k - 1] * obj->rho;
        }
      }
    } break;
  }

  return val;
}

/*
 * Arguments    : float workspace[5]
 *                int nVar
 *                const float grad[5]
 *                int mIneq
 *                int mEq
 *                const int finiteFixed[5]
 *                int mFixed
 *                const int finiteLB[5]
 *                int mLB
 *                const int finiteUB[5]
 *                int mUB
 *                const float lambda[9]
 * Return Type  : void
 */
__device__ void computeGradLag(float workspace[5], int nVar,
                               const float grad[5], int mIneq, int mEq,
                               const int finiteFixed[5], int mFixed,
                               const int finiteLB[5], int mLB,
                               const int finiteUB[5], int mUB,
                               const float lambda[9]) {
  int b_i;
  int iL0;
  int idx;
  if (nVar - 1 >= 0) {
    (void)memcpy(&workspace[0], &grad[0], (unsigned int)nVar * sizeof(float));
  }

  for (idx = 0; idx < mFixed; idx++) {
    b_i = finiteFixed[idx];
    workspace[b_i - 1] += lambda[idx];
  }

  iL0 = (mFixed + mEq) + mIneq;
  for (idx = 0; idx < mLB; idx++) {
    b_i = finiteLB[idx];
    workspace[b_i - 1] -= lambda[iL0 + idx];
  }

  if (mLB - 1 >= 0) {
    iL0 += mLB;
  }

  for (idx = 0; idx < mUB; idx++) {
    b_i = finiteUB[idx];
    workspace[b_i - 1] += lambda[iL0 + idx];
  }
}

/*
 * Arguments    : j_struct_T *obj
 *                const float b_H[16]
 *                const float f[5]
 *                const float x[5]
 * Return Type  : void
 */
__device__ void computeGrad_StoreHx(j_struct_T *obj, const float b_H[16],
                                    const float f[5], const float x[5]) {
  int ia;
  int ix;
  switch (obj->objtype) {
    case 5: {
      int b_i;
      b_i = obj->b_nvar;
      if (b_i - 2 >= 0) {
        (void)memset(&obj->grad[0], 0,
                     (unsigned int)(int)(b_i - 1) * sizeof(float));
      }

      obj->grad[obj->b_nvar - 1] = obj->gammaScalar;
    } break;

    case 3: {
      int b_i;
      int iy;
      int m_tmp;
      m_tmp = obj->b_nvar - 1;
      iy = obj->b_nvar;
      if (obj->b_nvar != 0) {
        int iac;
        if (m_tmp >= 0) {
          (void)memset(&obj->Hx[0], 0,
                       (unsigned int)(int)(m_tmp + 1) * sizeof(float));
        }

        ix = 0;
        b_i = obj->b_nvar * m_tmp + 1;
        iac = 1;
        while (((iy > 0) && (iac <= b_i)) || ((iy < 0) && (iac >= b_i))) {
          int i1;
          i1 = iac + m_tmp;
          for (ia = iac; ia <= i1; ia++) {
            int i2;
            i2 = ia - iac;
            obj->Hx[i2] += b_H[ia - 1] * x[ix];
          }

          ix++;
          iac += iy;
        }
      }

      b_i = obj->b_nvar;
      if (b_i - 1 >= 0) {
        (void)memcpy(&obj->grad[0], &obj->Hx[0],
                     (unsigned int)b_i * sizeof(float));
      }

      if (obj->hasLinear && (obj->b_nvar >= 1)) {
        for (ix = 0; ix <= m_tmp; ix++) {
          obj->grad[ix] += f[ix];
        }
      }
    } break;

    case 4: {
      int b_i;
      int i1;
      int iy;
      int m_tmp;
      int maxRegVar;
      maxRegVar = obj->maxVar - 1;
      m_tmp = obj->b_nvar - 1;
      iy = obj->b_nvar;
      if (obj->b_nvar != 0) {
        int iac;
        if (m_tmp >= 0) {
          (void)memset(&obj->Hx[0], 0,
                       (unsigned int)(int)(m_tmp + 1) * sizeof(float));
        }

        ix = 0;
        b_i = obj->b_nvar * (obj->b_nvar - 1) + 1;
        iac = 1;
        while (((iy > 0) && (iac <= b_i)) || ((iy < 0) && (iac >= b_i))) {
          i1 = iac + m_tmp;
          for (ia = iac; ia <= i1; ia++) {
            int i2;
            i2 = ia - iac;
            obj->Hx[i2] += b_H[ia - 1] * x[ix];
          }

          ix++;
          iac += iy;
        }
      }

      b_i = obj->b_nvar + 1;
      for (ix = b_i; ix <= maxRegVar; ix++) {
        obj->Hx[ix - 1] = obj->beta * x[ix - 1];
      }

      if (maxRegVar - 1 >= 0) {
        (void)memcpy(&obj->grad[0], &obj->Hx[0],
                     (unsigned int)maxRegVar * sizeof(float));
      }

      if (obj->hasLinear && (obj->b_nvar >= 1)) {
        for (ix = 0; ix <= m_tmp; ix++) {
          obj->grad[ix] += f[ix];
        }
      }

      ix = (obj->maxVar - obj->b_nvar) - 1;
      if (ix >= 1) {
        iy = obj->b_nvar;
        b_i = ix - 1;
        for (ix = 0; ix <= b_i; ix++) {
          i1 = iy + ix;
          obj->grad[i1] += obj->rho;
        }
      }
    } break;
  }
}

/*
 * Arguments    : int nVar
 *                int mConstr
 *                c_struct_T *QRManager
 *                const float ATwset[45]
 *                int ldA
 *                const float grad[5]
 *                float lambdaLSQ[9]
 *                float workspace[45]
 * Return Type  : void
 */
__device__ void computeLambdaLSQ(int nVar, int mConstr, c_struct_T *QRManager,
                                 const float ATwset[45], int ldA,
                                 const float grad[5], float lambdaLSQ[9],
                                 float workspace[45]) {
  float b_c;
  int fullRank_R;
  int iQR_diag;
  int ia;
  int iac;
  int ix;
  int rankR;
  bool guard1;
  if (mConstr - 1 >= 0) {
    (void)memset(&lambdaLSQ[0], 0, (unsigned int)mConstr * sizeof(float));
  }

  iQR_diag = nVar * mConstr;
  guard1 = false;
  if (iQR_diag > 0) {
    for (rankR = 0; rankR < mConstr; rankR++) {
      iQR_diag = ldA * rankR;
      ix = QRManager->ldq * rankR;
      for (fullRank_R = 0; fullRank_R < nVar; fullRank_R++) {
        QRManager->QR[ix + fullRank_R] = ATwset[iQR_diag + fullRank_R];
      }
    }

    guard1 = true;
  } else if (iQR_diag == 0) {
    QRManager->mrows = nVar;
    QRManager->ncols = mConstr;
    QRManager->minRowCol = 0;
  } else {
    guard1 = true;
  }

  if (guard1) {
    QRManager->usedPivoting = true;
    QRManager->mrows = nVar;
    QRManager->ncols = mConstr;
    if (nVar <= mConstr) {
      iQR_diag = nVar;
    } else {
      iQR_diag = mConstr;
    }

    QRManager->minRowCol = iQR_diag;
    xgeqp3(QRManager->QR, nVar, mConstr, QRManager->jpvt, QRManager->tau);
  }

  computeQ_(QRManager, QRManager->mrows);
  ix = QRManager->ldq;
  if (nVar != 0) {
    if (nVar - 1 >= 0) {
      (void)memset(&workspace[0], 0, (unsigned int)nVar * sizeof(float));
    }

    fullRank_R = 0;
    iQR_diag = QRManager->ldq * (nVar - 1) + 1;
    iac = 1;
    while (((ix > 0) && (iac <= iQR_diag)) || ((ix < 0) && (iac >= iQR_diag))) {
      b_c = 0.0F;
      rankR = (iac + nVar) - 1;
      for (ia = iac; ia <= rankR; ia++) {
        b_c += QRManager->Q[ia - 1] * grad[ia - iac];
      }

      workspace[fullRank_R] -= b_c;
      fullRank_R++;
      iac += ix;
    }
  }

  if (nVar >= mConstr) {
    iQR_diag = nVar;
  } else {
    iQR_diag = mConstr;
  }

  b_c = fabsf(QRManager->QR[0]) *
        fminf(0.000345266977F, (float)iQR_diag * 1.1920929E-7F);
  if (nVar <= mConstr) {
    fullRank_R = nVar;
  } else {
    fullRank_R = mConstr;
  }

  rankR = 0;
  iQR_diag = 0;
  while ((rankR < fullRank_R) && (fabsf(QRManager->QR[iQR_diag]) > b_c)) {
    rankR++;
    iQR_diag = (iQR_diag + QRManager->ldq) + 1;
  }

  if (rankR != 0) {
    for (iac = rankR; iac >= 1; iac--) {
      iQR_diag = (iac + (iac - 1) * QRManager->ldq) - 1;
      workspace[iac - 1] /= QRManager->QR[iQR_diag];
      for (ia = 0; ia <= iac - 2; ia++) {
        ix = (iac - ia) - 2;
        workspace[ix] -=
            workspace[iac - 1] * QRManager->QR[(iQR_diag - ia) - 1];
      }
    }
  }

  if (mConstr <= fullRank_R) {
    fullRank_R = mConstr;
  }

  for (rankR = 0; rankR < fullRank_R; rankR++) {
    lambdaLSQ[QRManager->jpvt[rankR] - 1] = workspace[rankR];
  }
}

/*
 * Arguments    : const float x[4]
 *                const int finiteLB[5]
 *                int mLB
 *                const float lb[4]
 *                const int finiteUB[5]
 *                int mUB
 *                const float ub[4]
 * Return Type  : float
 */
__device__ float computePrimalFeasError(const float x[4], const int finiteLB[5],
                                        int mLB, const float lb[4],
                                        const int finiteUB[5], int mUB,
                                        const float ub[4]) {
  float feasError;
  int b_i;
  int idx;
  feasError = 0.0F;
  for (idx = 0; idx < mLB; idx++) {
    b_i = finiteLB[idx];
    feasError = fmaxf(feasError, lb[b_i - 1] - x[b_i - 1]);
  }

  for (idx = 0; idx < mUB; idx++) {
    b_i = finiteUB[idx];
    feasError = fmaxf(feasError, x[b_i - 1] - ub[b_i - 1]);
  }

  return feasError;
}

/*
 * Arguments    : c_struct_T *obj
 *                int nrows
 * Return Type  : void
 */
__device__ void computeQ_(c_struct_T *obj, int nrows) {
  float work[9];
  int b_lda;
  int c_i;
  int iQR0;
  int ia;
  int idx;
  int lastc;
  int m;
  int n;
  lastc = obj->minRowCol;
  for (idx = 0; idx < lastc; idx++) {
    iQR0 = obj->ldq * idx + idx;
    n = obj->mrows - idx;
    if (n - 2 >= 0) {
      (void)memcpy(
          &obj->Q[iQR0 + 1], &obj->QR[iQR0 + 1],
          (unsigned int)(int)(((n + iQR0) - iQR0) - 1) * sizeof(float));
    }
  }

  m = obj->mrows;
  b_lda = obj->ldq;
  if (nrows >= 1) {
    int b_i;
    int itau;
    b_i = nrows - 1;
    for (n = lastc; n <= b_i; n++) {
      ia = n * b_lda;
      idx = m - 1;
      if (idx >= 0) {
        (void)memset(
            &obj->Q[ia], 0,
            (unsigned int)(int)(((idx + ia) - ia) + 1) * sizeof(float));
      }

      obj->Q[ia + n] = 1.0F;
    }

    itau = obj->minRowCol - 1;
    for (c_i = 0; c_i < 9; c_i++) {
      work[c_i] = 0.0F;
    }

    for (c_i = obj->minRowCol; c_i >= 1; c_i--) {
      int iaii;
      iaii = (c_i + (c_i - 1) * b_lda) - 1;
      if (c_i < nrows) {
        int jA;
        int lastv;
        obj->Q[iaii] = 1.0F;
        jA = (iaii + b_lda) + 1;
        if (obj->tau[itau] != 0.0F) {
          bool exitg2;
          lastv = m - c_i;
          iQR0 = (iaii + m) - c_i;
          while ((lastv + 1 > 0) && (obj->Q[iQR0] == 0.0F)) {
            lastv--;
            iQR0--;
          }

          lastc = (nrows - c_i) - 1;
          exitg2 = false;
          while ((!exitg2) && (lastc + 1 > 0)) {
            int exitg1;
            iQR0 = jA + lastc * b_lda;
            ia = iQR0;
            do {
              exitg1 = 0;
              if (ia <= iQR0 + lastv) {
                if (obj->Q[ia - 1] != 0.0F) {
                  exitg1 = 1;
                } else {
                  ia++;
                }
              } else {
                lastc--;
                exitg1 = 2;
              }
            } while (exitg1 == 0);

            if (exitg1 == 1) {
              exitg2 = true;
            }
          }
        } else {
          lastv = -1;
          lastc = -1;
        }

        if (lastv + 1 > 0) {
          float b_c;
          if (lastc + 1 != 0) {
            if (lastc >= 0) {
              (void)memset(&work[0], 0,
                           (unsigned int)(int)(lastc + 1) * sizeof(float));
            }

            iQR0 = 0;
            b_i = jA + b_lda * lastc;
            n = jA;
            while (((b_lda > 0) && (n <= b_i)) || ((b_lda < 0) && (n >= b_i))) {
              b_c = 0.0F;
              idx = n + lastv;
              for (ia = n; ia <= idx; ia++) {
                b_c += obj->Q[ia - 1] * obj->Q[(iaii + ia) - n];
              }

              work[iQR0] += b_c;
              iQR0++;
              n += b_lda;
            }
          }

          if (-obj->tau[itau] != 0.0F) {
            for (n = 0; n <= lastc; n++) {
              b_c = work[n];
              if (b_c != 0.0F) {
                b_c *= -obj->tau[itau];
                b_i = lastv + jA;
                for (iQR0 = jA; iQR0 <= b_i; iQR0++) {
                  obj->Q[iQR0 - 1] += obj->Q[(iaii + iQR0) - jA] * b_c;
                }
              }

              jA += b_lda;
            }
          }
        }
      }

      if (c_i < m) {
        iQR0 = iaii + 2;
        b_i = ((iaii + m) - c_i) + 1;
        for (lastc = iQR0; lastc <= b_i; lastc++) {
          obj->Q[lastc - 1] *= -obj->tau[itau];
        }
      }

      obj->Q[iaii] = 1.0F - obj->tau[itau];
      for (n = 0; n <= c_i - 2; n++) {
        obj->Q[(iaii - n) - 1] = 0.0F;
      }

      itau--;
    }
  }
}

/*
 * Arguments    : const float b_H[16]
 *                d_struct_T *solution
 *                b_struct_T *memspace
 *                const c_struct_T *b_qrmanager
 *                k_struct_T *b_cholmanager
 *                const j_struct_T *objective
 *                bool alwaysPositiveDef
 * Return Type  : void
 */
__device__ void compute_deltax(const float b_H[16], d_struct_T *solution,
                               b_struct_T *memspace,
                               const c_struct_T *b_qrmanager,
                               k_struct_T *b_cholmanager,
                               const j_struct_T *objective,
                               bool alwaysPositiveDef) {
  int c_i;
  int ia;
  int idx_col;
  int idx_row;
  int ix;
  int mNull_tmp;
  int nVar_tmp;
  nVar_tmp = b_qrmanager->mrows - 1;
  mNull_tmp = b_qrmanager->mrows - b_qrmanager->ncols;
  if (mNull_tmp <= 0) {
    if (nVar_tmp >= 0) {
      (void)memset(&solution->searchDir[0], 0,
                   (unsigned int)(int)(nVar_tmp + 1) * sizeof(float));
    }
  } else {
    for (ix = 0; ix <= nVar_tmp; ix++) {
      solution->searchDir[ix] = -objective->grad[ix];
    }

    if (b_qrmanager->ncols <= 0) {
      switch (objective->objtype) {
        case 5:
          break;

        case 3: {
          float SCALED_REG_PRIMAL;
          int nVars;
          if (alwaysPositiveDef) {
            b_cholmanager->ndims = b_qrmanager->mrows;
            for (ix = 0; ix <= nVar_tmp; ix++) {
              idx_row = (nVar_tmp + 1) * ix;
              idx_col = b_cholmanager->ldm * ix;
              for (c_i = 0; c_i <= nVar_tmp; c_i++) {
                b_cholmanager->FMat[idx_col + c_i] = b_H[idx_row + c_i];
              }
            }

            b_cholmanager->info = xpotrf(
                b_qrmanager->mrows, b_cholmanager->FMat, b_cholmanager->ldm);
          } else {
            SCALED_REG_PRIMAL = 0.000345266977F * b_cholmanager->scaleFactor *
                                (float)b_qrmanager->mrows;
            b_cholmanager->ndims = b_qrmanager->mrows;
            for (ix = 0; ix <= nVar_tmp; ix++) {
              idx_row = b_qrmanager->mrows * ix;
              idx_col = b_cholmanager->ldm * ix;
              for (c_i = 0; c_i <= nVar_tmp; c_i++) {
                b_cholmanager->FMat[idx_col + c_i] = b_H[idx_row + c_i];
              }
            }

            nVars = ixamax(b_qrmanager->mrows, b_cholmanager->FMat,
                           b_cholmanager->ldm + 1) -
                    1;
            b_cholmanager->regTol_ = fmaxf(
                fabsf(b_cholmanager->FMat[nVars + b_cholmanager->ldm * nVars]) *
                    1.1920929E-7F,
                fabsf(SCALED_REG_PRIMAL));
            fullColLDL2_(b_cholmanager, b_qrmanager->mrows, SCALED_REG_PRIMAL);
            if (b_cholmanager->ConvexCheck) {
              ix = 0;
              int exitg1;
              do {
                exitg1 = 0;
                if (ix <= nVar_tmp) {
                  if (b_cholmanager->FMat[ix + b_cholmanager->ldm * ix] <=
                      0.0F) {
                    b_cholmanager->info = -ix - 1;
                    exitg1 = 1;
                  } else {
                    ix++;
                  }
                } else {
                  b_cholmanager->ConvexCheck = false;
                  exitg1 = 1;
                }
              } while (exitg1 == 0);
            }
          }

          if (b_cholmanager->info != 0) {
            solution->state = -6;
          } else if (alwaysPositiveDef) {
            solve(b_cholmanager, solution->searchDir);
          } else {
            int b_i;
            idx_col = b_cholmanager->ndims - 2;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = 0; idx_row <= idx_col + 1; idx_row++) {
                nVars = idx_row + idx_row * b_cholmanager->ldm;
                b_i = idx_col - idx_row;
                for (c_i = 0; c_i <= b_i; c_i++) {
                  ix = (idx_row + c_i) + 1;
                  solution->searchDir[ix] -=
                      solution->searchDir[idx_row] *
                      b_cholmanager->FMat[(nVars + c_i) + 1];
                }
              }
            }

            b_i = b_cholmanager->ndims;
            for (ix = 0; ix < b_i; ix++) {
              solution->searchDir[ix] /=
                  b_cholmanager->FMat[ix + b_cholmanager->ldm * ix];
            }

            idx_col = b_cholmanager->ndims;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = idx_col; idx_row >= 1; idx_row--) {
                nVars = (idx_row - 1) * b_cholmanager->ldm;
                SCALED_REG_PRIMAL = solution->searchDir[idx_row - 1];
                b_i = idx_row + 1;
                for (c_i = idx_col; c_i >= b_i; c_i--) {
                  SCALED_REG_PRIMAL -= b_cholmanager->FMat[(nVars + c_i) - 1] *
                                       solution->searchDir[c_i - 1];
                }

                solution->searchDir[idx_row - 1] = SCALED_REG_PRIMAL;
              }
            }
          }
        } break;

        case 4: {
          if (alwaysPositiveDef) {
            int nVars;
            nVars = objective->b_nvar;
            b_cholmanager->ndims = objective->b_nvar;
            for (ix = 0; ix < nVars; ix++) {
              idx_row = nVars * ix;
              idx_col = b_cholmanager->ldm * ix;
              for (c_i = 0; c_i < nVars; c_i++) {
                b_cholmanager->FMat[idx_col + c_i] = b_H[idx_row + c_i];
              }
            }

            b_cholmanager->info = xpotrf(objective->b_nvar, b_cholmanager->FMat,
                                         b_cholmanager->ldm);
            if (b_cholmanager->info != 0) {
              solution->state = -6;
            } else {
              float SCALED_REG_PRIMAL;
              int b_i;
              solve(b_cholmanager, solution->searchDir);
              SCALED_REG_PRIMAL = 1.0F / objective->beta;
              idx_row = objective->b_nvar + 1;
              b_i = b_qrmanager->mrows;
              for (c_i = idx_row; c_i <= b_i; c_i++) {
                solution->searchDir[c_i - 1] *= SCALED_REG_PRIMAL;
              }
            }
          }
        } break;
      }
    } else {
      int nullStartIdx_tmp;
      nullStartIdx_tmp = b_qrmanager->ldq * b_qrmanager->ncols + 1;
      if (objective->objtype == 5) {
        for (ix = 0; ix < mNull_tmp; ix++) {
          memspace->workspace_float[ix] =
              -b_qrmanager
                   ->Q[nVar_tmp + b_qrmanager->ldq * (b_qrmanager->ncols + ix)];
        }

        idx_row = b_qrmanager->ldq;
        if (b_qrmanager->mrows != 0) {
          int b_i;
          int iac;
          if (nVar_tmp >= 0) {
            (void)memset(&solution->searchDir[0], 0,
                         (unsigned int)(int)(nVar_tmp + 1) * sizeof(float));
          }

          ix = 0;
          b_i = nullStartIdx_tmp + b_qrmanager->ldq * (mNull_tmp - 1);
          iac = nullStartIdx_tmp;
          while (((idx_row > 0) && (iac <= b_i)) ||
                 ((idx_row < 0) && (iac >= b_i))) {
            c_i = iac + nVar_tmp;
            for (ia = iac; ia <= c_i; ia++) {
              int nVars;
              nVars = ia - iac;
              solution->searchDir[nVars] +=
                  b_qrmanager->Q[ia - 1] * memspace->workspace_float[ix];
            }

            ix++;
            iac += idx_row;
          }
        }
      } else {
        float SCALED_REG_PRIMAL;
        int b_i;
        int nVars;
        if (objective->objtype == 3) {
          xgemm(b_qrmanager->mrows, mNull_tmp, b_qrmanager->mrows, b_H,
                b_qrmanager->mrows, b_qrmanager->Q, nullStartIdx_tmp,
                b_qrmanager->ldq, memspace->workspace_float);
          b_xgemm(mNull_tmp, mNull_tmp, b_qrmanager->mrows, b_qrmanager->Q,
                  nullStartIdx_tmp, b_qrmanager->ldq, memspace->workspace_float,
                  b_cholmanager->FMat, b_cholmanager->ldm);
        } else if (alwaysPositiveDef) {
          nVars = b_qrmanager->mrows;
          xgemm(objective->b_nvar, mNull_tmp, objective->b_nvar, b_H,
                objective->b_nvar, b_qrmanager->Q, nullStartIdx_tmp,
                b_qrmanager->ldq, memspace->workspace_float);
          b_i = objective->b_nvar + 1;
          for (idx_col = 0; idx_col < mNull_tmp; idx_col++) {
            for (idx_row = b_i; idx_row <= nVars; idx_row++) {
              memspace->workspace_float[(idx_row + 9 * idx_col) - 1] =
                  objective->beta *
                  b_qrmanager
                      ->Q[(idx_row + 9 * (idx_col + b_qrmanager->ncols)) - 1];
            }
          }

          b_xgemm(mNull_tmp, mNull_tmp, b_qrmanager->mrows, b_qrmanager->Q,
                  nullStartIdx_tmp, b_qrmanager->ldq, memspace->workspace_float,
                  b_cholmanager->FMat, b_cholmanager->ldm);
        } else {
          /* no actions */
        }

        if (alwaysPositiveDef) {
          b_cholmanager->ndims = mNull_tmp;
          b_cholmanager->info =
              xpotrf(mNull_tmp, b_cholmanager->FMat, b_cholmanager->ldm);
        } else {
          SCALED_REG_PRIMAL =
              1.49011612E-8F * b_cholmanager->scaleFactor * (float)mNull_tmp;
          b_cholmanager->ndims = mNull_tmp;
          nVars =
              ixamax(mNull_tmp, b_cholmanager->FMat, b_cholmanager->ldm + 1) -
              1;
          b_cholmanager->regTol_ = fmaxf(
              fabsf(b_cholmanager->FMat[nVars + b_cholmanager->ldm * nVars]) *
                  2.22044605E-16F,
              fabsf(SCALED_REG_PRIMAL));
          fullColLDL2_(b_cholmanager, mNull_tmp, SCALED_REG_PRIMAL);
          if (b_cholmanager->ConvexCheck) {
            ix = 0;
            int exitg1;
            do {
              exitg1 = 0;
              if (ix <= mNull_tmp - 1) {
                if (b_cholmanager->FMat[ix + b_cholmanager->ldm * ix] <= 0.0F) {
                  b_cholmanager->info = -ix - 1;
                  exitg1 = 1;
                } else {
                  ix++;
                }
              } else {
                b_cholmanager->ConvexCheck = false;
                exitg1 = 1;
              }
            } while (exitg1 == 0);
          }
        }

        if (b_cholmanager->info != 0) {
          solution->state = -6;
        } else {
          int iac;
          int lda_tmp;
          lda_tmp = b_qrmanager->ldq;
          if (b_qrmanager->mrows != 0) {
            (void)memset(&memspace->workspace_float[0], 0,
                         (unsigned int)mNull_tmp * sizeof(float));
            idx_col = 0;
            b_i = nullStartIdx_tmp + b_qrmanager->ldq * (mNull_tmp - 1);
            iac = nullStartIdx_tmp;
            while (((lda_tmp > 0) && (iac <= b_i)) ||
                   ((lda_tmp < 0) && (iac >= b_i))) {
              SCALED_REG_PRIMAL = 0.0F;
              c_i = iac + nVar_tmp;
              for (ia = iac; ia <= c_i; ia++) {
                SCALED_REG_PRIMAL +=
                    b_qrmanager->Q[ia - 1] * objective->grad[ia - iac];
              }

              memspace->workspace_float[idx_col] -= SCALED_REG_PRIMAL;
              idx_col++;
              iac += lda_tmp;
            }
          }

          if (alwaysPositiveDef) {
            idx_col = b_cholmanager->ndims;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = 0; idx_row < idx_col; idx_row++) {
                nVars = idx_row * b_cholmanager->ldm;
                SCALED_REG_PRIMAL = memspace->workspace_float[idx_row];
                for (c_i = 0; c_i < idx_row; c_i++) {
                  SCALED_REG_PRIMAL -= b_cholmanager->FMat[nVars + c_i] *
                                       memspace->workspace_float[c_i];
                }

                memspace->workspace_float[idx_row] =
                    SCALED_REG_PRIMAL / b_cholmanager->FMat[nVars + idx_row];
              }
            }

            idx_col = b_cholmanager->ndims;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = idx_col; idx_row >= 1; idx_row--) {
                nVars = (idx_row + (idx_row - 1) * b_cholmanager->ldm) - 1;
                memspace->workspace_float[idx_row - 1] /=
                    b_cholmanager->FMat[nVars];
                for (c_i = 0; c_i <= idx_row - 2; c_i++) {
                  ix = (idx_row - c_i) - 2;
                  memspace->workspace_float[ix] -=
                      memspace->workspace_float[idx_row - 1] *
                      b_cholmanager->FMat[(nVars - c_i) - 1];
                }
              }
            }
          } else {
            idx_col = b_cholmanager->ndims - 2;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = 0; idx_row <= idx_col + 1; idx_row++) {
                nVars = idx_row + idx_row * b_cholmanager->ldm;
                b_i = idx_col - idx_row;
                for (c_i = 0; c_i <= b_i; c_i++) {
                  ix = (idx_row + c_i) + 1;
                  memspace->workspace_float[ix] -=
                      memspace->workspace_float[idx_row] *
                      b_cholmanager->FMat[(nVars + c_i) + 1];
                }
              }
            }

            b_i = b_cholmanager->ndims;
            for (ix = 0; ix < b_i; ix++) {
              memspace->workspace_float[ix] /=
                  b_cholmanager->FMat[ix + b_cholmanager->ldm * ix];
            }

            idx_col = b_cholmanager->ndims;
            if (b_cholmanager->ndims != 0) {
              for (idx_row = idx_col; idx_row >= 1; idx_row--) {
                nVars = (idx_row - 1) * b_cholmanager->ldm;
                SCALED_REG_PRIMAL = memspace->workspace_float[idx_row - 1];
                b_i = idx_row + 1;
                for (c_i = idx_col; c_i >= b_i; c_i--) {
                  SCALED_REG_PRIMAL -= b_cholmanager->FMat[(nVars + c_i) - 1] *
                                       memspace->workspace_float[c_i - 1];
                }

                memspace->workspace_float[idx_row - 1] = SCALED_REG_PRIMAL;
              }
            }
          }

          if (b_qrmanager->mrows != 0) {
            if (nVar_tmp >= 0) {
              (void)memset(&solution->searchDir[0], 0,
                           (unsigned int)(int)(nVar_tmp + 1) * sizeof(float));
            }

            ix = 0;
            b_i = nullStartIdx_tmp + b_qrmanager->ldq * (mNull_tmp - 1);
            iac = nullStartIdx_tmp;
            while (((lda_tmp > 0) && (iac <= b_i)) ||
                   ((lda_tmp < 0) && (iac >= b_i))) {
              c_i = iac + nVar_tmp;
              for (ia = iac; ia <= c_i; ia++) {
                nVars = ia - iac;
                solution->searchDir[nVars] +=
                    b_qrmanager->Q[ia - 1] * memspace->workspace_float[ix];
              }

              ix++;
              iac += lda_tmp;
            }
          }
        }
      }
    }
  }
}

/*
 * Arguments    : int x[9]
 *                int xLen
 *                int workspace[9]
 *                int xMin
 *                int xMax
 * Return Type  : void
 */
__device__ void countsort(int x[9], int xLen, int workspace[9], int xMin,
                          int xMax) {
  int idx;
  int idxFill;
  if ((xLen > 1) && (xMax > xMin)) {
    int idxEnd;
    int idxStart;
    int maxOffset;
    idxStart = xMax - xMin;
    if (idxStart >= 0) {
      (void)memset(&workspace[0], 0,
                   (unsigned int)(int)(idxStart + 1) * sizeof(int));
    }

    maxOffset = idxStart - 1;
    for (idx = 0; idx < xLen; idx++) {
      idxStart = x[idx] - xMin;
      workspace[idxStart]++;
    }

    for (idx = 2; idx <= maxOffset + 2; idx++) {
      workspace[idx - 1] += workspace[idx - 2];
    }

    idxStart = 1;
    idxEnd = workspace[0];
    for (idx = 0; idx <= maxOffset; idx++) {
      for (idxFill = idxStart; idxFill <= idxEnd; idxFill++) {
        x[idxFill - 1] = idx + xMin;
      }

      idxStart = workspace[idx] + 1;
      idxEnd = workspace[idx + 1];
    }

    for (idx = idxStart; idx <= idxEnd; idx++) {
      x[idx - 1] = xMax;
    }
  }
}

/*
 * Arguments    : c_struct_T *obj
 *                int idx
 * Return Type  : void
 */
__device__ void deleteColMoveEnd(c_struct_T *obj, int idx) {
  float s;
  float temp;
  int b_i;
  int b_k;
  int k;
  if (obj->usedPivoting) {
    b_i = 1;
    while ((b_i <= obj->ncols) && (obj->jpvt[b_i - 1] != idx)) {
      b_i++;
    }

    idx = b_i;
  }

  if (idx >= obj->ncols) {
    obj->ncols--;
  } else {
    int c_i;
    int ix;
    c_i = obj->ncols - 1;
    obj->jpvt[idx - 1] = obj->jpvt[c_i];
    b_i = obj->minRowCol;
    for (k = 0; k < b_i; k++) {
      obj->QR[k + obj->ldq * (idx - 1)] = obj->QR[k + obj->ldq * c_i];
    }

    obj->ncols = c_i;
    ix = obj->mrows;
    b_i = obj->ncols;
    if (ix <= b_i) {
      b_i = ix;
    }

    obj->minRowCol = b_i;
    if (idx < obj->mrows) {
      float b_c;
      int endIdx;
      int idxRotGCol;
      int n;
      int temp_tmp;
      ix = obj->mrows - 1;
      endIdx = obj->ncols;
      if (ix <= endIdx) {
        endIdx = ix;
      }

      k = endIdx;
      idxRotGCol = obj->ldq * (idx - 1);
      while (k >= idx) {
        c_i = k + idxRotGCol;
        temp = obj->QR[c_i];
        b_c = xrotg(&obj->QR[c_i - 1], &temp, &s);
        obj->QR[c_i] = temp;
        c_i = obj->ldq * (k - 1);
        obj->QR[k + c_i] = 0.0F;
        b_i = k + obj->ldq * idx;
        n = obj->ncols - idx;
        if (n >= 1) {
          ix = b_i - 1;
          for (b_k = 0; b_k < n; b_k++) {
            temp = b_c * obj->QR[ix] + s * obj->QR[b_i];
            obj->QR[b_i] = b_c * obj->QR[b_i] - s * obj->QR[ix];
            obj->QR[ix] = temp;
            b_i += obj->ldq;
            ix += obj->ldq;
          }
        }

        b_i = obj->ldq + c_i;
        n = obj->mrows;
        for (b_k = 0; b_k < n; b_k++) {
          ix = b_i + b_k;
          temp_tmp = c_i + b_k;
          temp = b_c * obj->Q[temp_tmp] + s * obj->Q[ix];
          obj->Q[ix] = b_c * obj->Q[ix] - s * obj->Q[temp_tmp];
          obj->Q[temp_tmp] = temp;
        }

        k--;
      }

      c_i = idx + 1;
      for (k = c_i; k <= endIdx; k++) {
        idxRotGCol = obj->ldq * (k - 1);
        b_i = k + idxRotGCol;
        temp = obj->QR[b_i];
        b_c = xrotg(&obj->QR[b_i - 1], &temp, &s);
        obj->QR[b_i] = temp;
        b_i = k * (obj->ldq + 1);
        n = obj->ncols - k;
        if (n >= 1) {
          ix = b_i - 1;
          for (b_k = 0; b_k < n; b_k++) {
            temp = b_c * obj->QR[ix] + s * obj->QR[b_i];
            obj->QR[b_i] = b_c * obj->QR[b_i] - s * obj->QR[ix];
            obj->QR[ix] = temp;
            b_i += obj->ldq;
            ix += obj->ldq;
          }
        }

        b_i = obj->ldq + idxRotGCol;
        n = obj->mrows;
        for (b_k = 0; b_k < n; b_k++) {
          ix = b_i + b_k;
          temp_tmp = idxRotGCol + b_k;
          temp = b_c * obj->Q[temp_tmp] + s * obj->Q[ix];
          obj->Q[ix] = b_c * obj->Q[ix] - s * obj->Q[temp_tmp];
          obj->Q[temp_tmp] = temp;
        }
      }
    }
  }
}

/*
 * Arguments    : int numerator
 * Return Type  : int
 */
__device__ int div_nde_s32_floor(int numerator) {
  int b_i;
  if ((numerator < 0) && (numerator % 9 != 0)) {
    b_i = -1;
  } else {
    b_i = 0;
  }

  return numerator / 9 + b_i;
}

/*
 * Arguments    : const float b_H[16]
 *                const float f[5]
 *                d_struct_T *solution
 *                b_struct_T *memspace
 *                e_struct_T *b_workingset
 *                c_struct_T *b_qrmanager
 *                k_struct_T *b_cholmanager
 *                j_struct_T *objective
 *                const g_struct_T options
 *                g_struct_T runTimeOptions
 * Return Type  : void
 */
__device__ void driver(const float b_H[16], const float f[5],
                       d_struct_T *solution, b_struct_T *memspace,
                       e_struct_T *b_workingset, c_struct_T *b_qrmanager,
                       k_struct_T *b_cholmanager, j_struct_T *objective,
                       const g_struct_T options, g_struct_T runTimeOptions) {
  int idxStartIneq;
  int idx_global;
  int mConstr;
  int nVar;
  bool guard1;
  solution->iterations = 0;
  runTimeOptions.RemainFeasible = (options.PricingTolerance <= 0.0F);
  nVar = b_workingset->nVar - 1;
  guard1 = false;
  if (b_workingset->probType == 3) {
    mConstr = b_workingset->sizes[0];
    for (idxStartIneq = 0; idxStartIneq < mConstr; idxStartIneq++) {
      solution->xstar[b_workingset->indexFixed[idxStartIneq] - 1] =
          b_workingset->ub[b_workingset->indexFixed[idxStartIneq] - 1];
    }

    mConstr = b_workingset->sizes[3];
    for (idxStartIneq = 0; idxStartIneq < mConstr; idxStartIneq++) {
      if (b_workingset
              ->isActiveConstr[(b_workingset->isActiveIdx[3] + idxStartIneq) -
                               1]) {
        solution->xstar[b_workingset->indexLB[idxStartIneq] - 1] =
            -b_workingset->lb[b_workingset->indexLB[idxStartIneq] - 1];
      }
    }

    mConstr = b_workingset->sizes[4];
    for (idxStartIneq = 0; idxStartIneq < mConstr; idxStartIneq++) {
      if (b_workingset
              ->isActiveConstr[(b_workingset->isActiveIdx[4] + idxStartIneq) -
                               1]) {
        solution->xstar[b_workingset->indexUB[idxStartIneq] - 1] =
            b_workingset->ub[b_workingset->indexUB[idxStartIneq] - 1];
      }
    }

    PresolveWorkingSet(solution, memspace, b_workingset, b_qrmanager, &options);
    if (solution->state >= 0) {
      guard1 = true;
    }
  } else {
    solution->state = 82;
    guard1 = true;
  }

  if (guard1) {
    float maxConstr_new;
    solution->iterations = 0;
    solution->maxConstr =
        b_maxConstraintViolation(b_workingset, solution->xstar);
    maxConstr_new =
        options.ConstraintTolerance * runTimeOptions.ConstrRelTolFactor;
    if (solution->maxConstr > maxConstr_new) {
      int PROBTYPE_ORIG;
      int c_nVar;
      int idxEndIneq_tmp_tmp;
      int nVarP1;
      PROBTYPE_ORIG = b_workingset->probType;
      c_nVar = b_workingset->nVar;
      nVarP1 = b_workingset->nVar;
      solution->xstar[b_workingset->nVar] = solution->maxConstr + 1.0F;
      if (b_workingset->probType == 3) {
        mConstr = 1;
      } else {
        mConstr = 4;
      }

      setProblemType(b_workingset, mConstr);
      mConstr = b_workingset->nWConstr[0] + b_workingset->nWConstr[1];
      idxStartIneq = mConstr + 1;
      idxEndIneq_tmp_tmp = b_workingset->nActiveConstr;
      for (idx_global = idxStartIneq; idx_global <= idxEndIneq_tmp_tmp;
           idx_global++) {
        b_workingset->isActiveConstr
            [(b_workingset->isActiveIdx[b_workingset->Wid[idx_global - 1] - 1] +
              b_workingset->Wlocalidx[idx_global - 1]) -
             2] = false;
      }

      b_workingset->nWConstr[2] = 0;
      b_workingset->nWConstr[3] = 0;
      b_workingset->nWConstr[4] = 0;
      b_workingset->nActiveConstr = mConstr;
      objective->prev_objtype = objective->objtype;
      objective->prev_nvar = objective->b_nvar;
      objective->prev_hasLinear = objective->hasLinear;
      objective->objtype = 5;
      objective->b_nvar = nVarP1 + 1;
      objective->gammaScalar = 1.0F;
      objective->hasLinear = true;
      solution->fstar = computeFval(objective, memspace->workspace_float, b_H,
                                    f, solution->xstar);
      solution->state = 5;
      iterate(b_H, f, solution, memspace, b_workingset, b_qrmanager,
              b_cholmanager, objective, options.SolverName, 3.45266972E-6F,
              options.ConstraintTolerance, maxConstr_new,
              options.PricingTolerance, runTimeOptions);
      if (b_workingset->isActiveConstr
              [(b_workingset->isActiveIdx[3] + b_workingset->sizes[3]) - 2]) {
        bool exitg1;
        idxStartIneq = b_workingset->sizes[0] + b_workingset->sizes[1];
        exitg1 = false;
        while ((!exitg1) && (idxStartIneq + 1 <= b_workingset->nActiveConstr)) {
          if ((b_workingset->Wid[idxStartIneq] == 4) &&
              (b_workingset->Wlocalidx[idxStartIneq] ==
               b_workingset->sizes[3])) {
            removeConstr(b_workingset, idxStartIneq + 1);
            exitg1 = true;
          } else {
            idxStartIneq++;
          }
        }
      }

      mConstr = b_workingset->nActiveConstr;
      idxStartIneq = b_workingset->sizes[0] + b_workingset->sizes[1];
      while ((mConstr > idxStartIneq) && (mConstr > c_nVar)) {
        removeConstr(b_workingset, mConstr);
        mConstr--;
      }

      solution->maxConstr = solution->xstar[nVarP1];
      setProblemType(b_workingset, PROBTYPE_ORIG);
      objective->objtype = objective->prev_objtype;
      objective->b_nvar = objective->prev_nvar;
      objective->hasLinear = objective->prev_hasLinear;
      if (solution->state != 0) {
        solution->maxConstr =
            b_maxConstraintViolation(b_workingset, solution->xstar);
        if (solution->maxConstr > maxConstr_new) {
          mConstr = b_workingset->mConstrMax;
          if (mConstr - 1 >= 0) {
            (void)memset(&solution->lambda[0], 0,
                         (unsigned int)mConstr * sizeof(float));
          }

          solution->fstar = computeFval(objective, memspace->workspace_float,
                                        b_H, f, solution->xstar);
          solution->state = -2;
        } else {
          if (solution->maxConstr > 0.0F) {
            if (nVar >= 0) {
              (void)memcpy(&solution->searchDir[0], &solution->xstar[0],
                           (unsigned int)(int)(nVar + 1) * sizeof(float));
            }

            PresolveWorkingSet(solution, memspace, b_workingset, b_qrmanager,
                               &options);
            maxConstr_new =
                b_maxConstraintViolation(b_workingset, solution->xstar);
            if (maxConstr_new >= solution->maxConstr) {
              solution->maxConstr = maxConstr_new;
              if (nVar >= 0) {
                (void)memcpy(&solution->xstar[0], &solution->searchDir[0],
                             (unsigned int)(int)(nVar + 1) * sizeof(float));
              }
            }
          }

          iterate(b_H, f, solution, memspace, b_workingset, b_qrmanager,
                  b_cholmanager, objective, options.SolverName,
                  options.StepTolerance, options.ConstraintTolerance,
                  options.ObjectiveLimit, options.PricingTolerance,
                  runTimeOptions);
        }
      }
    } else {
      iterate(b_H, f, solution, memspace, b_workingset, b_qrmanager,
              b_cholmanager, objective, options.SolverName,
              options.StepTolerance, options.ConstraintTolerance,
              options.ObjectiveLimit, options.PricingTolerance, runTimeOptions);
    }
  }
}

/*
 * Arguments    : c_struct_T *obj
 *                const float b_A[45]
 *                int mrows
 *                int ncols
 *                int ldA
 * Return Type  : void
 */
__device__ void factorQR(c_struct_T *obj, const float b_A[45], int mrows,
                         int ncols, int ldA) {
  int b_i;
  int idx;
  int ix0;
  int k;
  bool guard1;
  ix0 = mrows * ncols;
  guard1 = false;
  if (ix0 > 0) {
    for (idx = 0; idx < ncols; idx++) {
      ix0 = ldA * idx;
      b_i = obj->ldq * idx;
      for (k = 0; k < mrows; k++) {
        obj->QR[b_i + k] = b_A[ix0 + k];
      }
    }

    guard1 = true;
  } else if (ix0 == 0) {
    obj->mrows = mrows;
    obj->ncols = ncols;
    obj->minRowCol = 0;
  } else {
    guard1 = true;
  }

  if (guard1) {
    obj->usedPivoting = false;
    obj->mrows = mrows;
    obj->ncols = ncols;
    for (idx = 0; idx < ncols; idx++) {
      obj->jpvt[idx] = idx + 1;
    }

    if (mrows <= ncols) {
      ix0 = mrows;
    } else {
      ix0 = ncols;
    }

    obj->minRowCol = ix0;
    for (b_i = 0; b_i < 9; b_i++) {
      obj->tau[b_i] = 0.0F;
    }

    if (ix0 >= 1) {
      qrf(obj->QR, mrows, ncols, ix0, obj->tau);
    }
  }
}

/*
 * Arguments    : float workspace[45]
 *                float xCurrent[5]
 *                const e_struct_T *b_workingset
 *                c_struct_T *b_qrmanager
 * Return Type  : bool
 */
__device__ bool feasibleX0ForWorkingSet(float workspace[45], float xCurrent[5],
                                        const e_struct_T *b_workingset,
                                        c_struct_T *b_qrmanager) {
  float b_B[45];
  int br;
  int c_i;
  int iAcol;
  int ic;
  int iy;
  int jBcol;
  int k;
  int mWConstr;
  int nVar;
  bool nonDegenerateWset;
  mWConstr = b_workingset->nActiveConstr;
  nVar = b_workingset->nVar;
  nonDegenerateWset = true;
  if (mWConstr != 0) {
    float b_c;
    int b_i;
    int i1;
    for (iAcol = 0; iAcol < mWConstr; iAcol++) {
      b_c = b_workingset->bwset[iAcol];
      workspace[iAcol] = b_c;
      workspace[iAcol + 9] = b_c;
    }

    iAcol = b_workingset->ldA;
    if ((nVar != 0) && (mWConstr != 0)) {
      iy = 0;
      b_i = b_workingset->ldA * (mWConstr - 1) + 1;
      jBcol = 1;
      while (((iAcol > 0) && (jBcol <= b_i)) ||
             ((iAcol < 0) && (jBcol >= b_i))) {
        b_c = 0.0F;
        i1 = (jBcol + nVar) - 1;
        for (br = jBcol; br <= i1; br++) {
          b_c += b_workingset->ATwset[br - 1] * xCurrent[br - jBcol];
        }

        workspace[iy] -= b_c;
        iy++;
        jBcol += iAcol;
      }
    }

    if (mWConstr >= nVar) {
      int ldq;
      for (iy = 0; iy < nVar; iy++) {
        iAcol = b_qrmanager->ldq * iy;
        for (jBcol = 0; jBcol < mWConstr; jBcol++) {
          b_qrmanager->QR[jBcol + iAcol] =
              b_workingset->ATwset[iy + b_workingset->ldA * jBcol];
        }
      }

      if (mWConstr * nVar == 0) {
        b_qrmanager->mrows = mWConstr;
        b_qrmanager->ncols = nVar;
        b_qrmanager->minRowCol = 0;
      } else {
        b_qrmanager->usedPivoting = false;
        b_qrmanager->mrows = mWConstr;
        b_qrmanager->ncols = nVar;
        for (iAcol = 0; iAcol < nVar; iAcol++) {
          b_qrmanager->jpvt[iAcol] = iAcol + 1;
        }

        if (mWConstr <= nVar) {
          b_i = mWConstr;
        } else {
          b_i = nVar;
        }

        b_qrmanager->minRowCol = b_i;
        for (c_i = 0; c_i < 9; c_i++) {
          b_qrmanager->tau[c_i] = 0.0F;
        }

        if (b_i >= 1) {
          qrf(b_qrmanager->QR, mWConstr, nVar, b_i, b_qrmanager->tau);
        }
      }

      computeQ_(b_qrmanager, b_qrmanager->mrows);
      ldq = b_qrmanager->ldq;
      (void)memcpy(&b_B[0], &workspace[0], 45U * sizeof(float));
      if (nVar != 0) {
        for (k = 0; k <= 9; k += 9) {
          b_i = k + 1;
          i1 = k + nVar;
          if (b_i <= i1) {
            (void)memset(&workspace[b_i + -1], 0,
                         (unsigned int)(int)((i1 - b_i) + 1) * sizeof(float));
          }
        }

        br = -1;
        for (k = 0; k <= 9; k += 9) {
          jBcol = -1;
          b_i = k + 1;
          i1 = k + nVar;
          for (ic = b_i; ic <= i1; ic++) {
            b_c = 0.0F;
            for (iy = 0; iy < mWConstr; iy++) {
              b_c += b_qrmanager->Q[(iy + jBcol) + 1] * b_B[(iy + br) + 1];
            }

            workspace[ic - 1] += b_c;
            jBcol += ldq;
          }

          br += 9;
        }
      }

      for (br = 0; br < 2; br++) {
        jBcol = 9 * br - 1;
        for (k = nVar; k >= 1; k--) {
          iy = ldq * (k - 1) - 1;
          b_i = k + jBcol;
          b_c = workspace[b_i];
          if (b_c != 0.0F) {
            workspace[b_i] = b_c / b_qrmanager->QR[k + iy];
            for (c_i = 0; c_i <= k - 2; c_i++) {
              i1 = (c_i + jBcol) + 1;
              workspace[i1] -= workspace[b_i] * b_qrmanager->QR[(c_i + iy) + 1];
            }
          }
        }
      }
    } else {
      int ldq;
      factorQR(b_qrmanager, b_workingset->ATwset, nVar, mWConstr,
               b_workingset->ldA);
      computeQ_(b_qrmanager, b_qrmanager->minRowCol);
      ldq = b_qrmanager->ldq;
      for (br = 0; br < 2; br++) {
        jBcol = 9 * br;
        for (c_i = 0; c_i < mWConstr; c_i++) {
          iAcol = ldq * c_i;
          iy = c_i + jBcol;
          b_c = workspace[iy];
          for (k = 0; k < c_i; k++) {
            b_c -= b_qrmanager->QR[k + iAcol] * workspace[k + jBcol];
          }

          workspace[iy] = b_c / b_qrmanager->QR[c_i + iAcol];
        }
      }

      (void)memcpy(&b_B[0], &workspace[0], 45U * sizeof(float));
      if (nVar != 0) {
        for (k = 0; k <= 9; k += 9) {
          b_i = k + 1;
          i1 = k + nVar;
          if (b_i <= i1) {
            (void)memset(&workspace[b_i + -1], 0,
                         (unsigned int)(int)((i1 - b_i) + 1) * sizeof(float));
          }
        }

        br = 0;
        for (k = 0; k <= 9; k += 9) {
          jBcol = -1;
          b_i = br + 1;
          i1 = br + mWConstr;
          for (c_i = b_i; c_i <= i1; c_i++) {
            iy = k + 1;
            iAcol = k + nVar;
            for (ic = iy; ic <= iAcol; ic++) {
              workspace[ic - 1] +=
                  b_B[c_i - 1] * b_qrmanager->Q[(jBcol + ic) - k];
            }

            jBcol += ldq;
          }

          br += 9;
        }
      }
    }

    iAcol = 0;
    int exitg1;
    do {
      exitg1 = 0;
      if (iAcol <= nVar - 1) {
        if ((fabsf(workspace[iAcol]) >= 3.402823466E+38F) ||
            (fabsf(workspace[iAcol + 9]) >= 3.402823466E+38F)) {
          nonDegenerateWset = false;
          exitg1 = 1;
        } else {
          iAcol++;
        }
      } else {
        float constrViolation_basicX;
        if (nVar >= 1) {
          iAcol = nVar - 1;
          for (k = 0; k <= iAcol; k++) {
            workspace[k] += xCurrent[k];
          }
        }

        b_c = maxConstraintViolation(b_workingset, workspace, 1);
        constrViolation_basicX =
            maxConstraintViolation(b_workingset, workspace, 10);
        if ((b_c <= 1.1920929E-7F) || (b_c < constrViolation_basicX)) {
          if (nVar - 1 >= 0) {
            (void)memcpy(&xCurrent[0], &workspace[0],
                         (unsigned int)nVar * sizeof(float));
          }
        } else if (nVar - 1 >= 0) {
          (void)memcpy(&xCurrent[0], &workspace[9],
                       (unsigned int)nVar * sizeof(float));
        } else {
          /* no actions */
        }

        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }

  return nonDegenerateWset;
}

/*
 * Arguments    : const float solution_xstar[5]
 *                const float solution_searchDir[5]
 *                const float workspace[45]
 *                int workingset_nVar
 *                const float workingset_lb[5]
 *                const float workingset_ub[5]
 *                const int workingset_indexLB[5]
 *                const int workingset_indexUB[5]
 *                const int b_workingset_sizes[5]
 *                const int b_workingset_isActiveIdx[6]
 *                const bool workingset_isActiveConstr[9]
 *                const int workingset_nWConstr[5]
 *                bool isPhaseOne
 *                float tolcon
 *                bool *newBlocking
 *                int *constrType
 *                int *constrIdx
 * Return Type  : float
 */
__device__ float feasibleratiotest(
    const float solution_xstar[5], const float solution_searchDir[5],
    const float workspace[45], int workingset_nVar,
    const float workingset_lb[5], const float workingset_ub[5],
    const int workingset_indexLB[5], const int workingset_indexUB[5],
    const int b_workingset_sizes[5], const int b_workingset_isActiveIdx[6],
    const bool workingset_isActiveConstr[9], const int workingset_nWConstr[5],
    bool isPhaseOne, float tolcon, bool *newBlocking, int *constrType,
    int *constrIdx) {
  float alpha;
  float alphaTemp;
  float denomTol;
  float phaseOneCorrectionP;
  float phaseOneCorrectionX;
  float pk_corrected;
  int idx;
  int totalIneq;
  int totalUB;
  totalIneq = b_workingset_sizes[2];
  totalUB = b_workingset_sizes[4];
  alpha = 1.0E+30F;
  *newBlocking = false;
  *constrType = 0;
  *constrIdx = 0;
  denomTol = 0.00011920929F * b_xnrm2(workingset_nVar, solution_searchDir);
  if (workingset_nWConstr[2] < b_workingset_sizes[2]) {
    for (idx = 0; idx < totalIneq; idx++) {
      pk_corrected = workspace[idx + 9];
      if ((pk_corrected > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[2] + idx) -
                                      1])) {
        alphaTemp = workspace[idx];
        alphaTemp = fminf(fabsf(alphaTemp), tolcon - alphaTemp) / pk_corrected;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 3;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }
      }
    }
  }

  if (workingset_nWConstr[3] < b_workingset_sizes[3]) {
    phaseOneCorrectionX =
        (float)(isPhaseOne ? 1.0F : 0.0F) * solution_xstar[workingset_nVar - 1];
    phaseOneCorrectionP = (float)(isPhaseOne ? 1.0F : 0.0F) *
                          solution_searchDir[workingset_nVar - 1];
    totalIneq = b_workingset_sizes[3];
    for (idx = 0; idx <= totalIneq - 2; idx++) {
      int b_i;
      b_i = workingset_indexLB[idx];
      pk_corrected = -solution_searchDir[b_i - 1] - phaseOneCorrectionP;
      if ((pk_corrected > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[3] + idx) -
                                      1])) {
        alphaTemp = (-solution_xstar[b_i - 1] - workingset_lb[b_i - 1]) -
                    phaseOneCorrectionX;
        alphaTemp = fminf(fabsf(alphaTemp), tolcon - alphaTemp) / pk_corrected;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 4;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }
      }
    }

    totalIneq = workingset_indexLB[b_workingset_sizes[3] - 1] - 1;
    pk_corrected = -solution_searchDir[totalIneq];
    if ((pk_corrected > denomTol) &&
        (!workingset_isActiveConstr
             [(b_workingset_isActiveIdx[3] + b_workingset_sizes[3]) - 2])) {
      alphaTemp = -solution_xstar[totalIneq] - workingset_lb[totalIneq];
      alphaTemp = fminf(fabsf(alphaTemp), tolcon - alphaTemp) / pk_corrected;
      if (alphaTemp < alpha) {
        alpha = alphaTemp;
        *constrType = 4;
        *constrIdx = b_workingset_sizes[3];
        *newBlocking = true;
      }
    }
  }

  if (workingset_nWConstr[4] < b_workingset_sizes[4]) {
    phaseOneCorrectionX =
        (float)(isPhaseOne ? 1.0F : 0.0F) * solution_xstar[workingset_nVar - 1];
    phaseOneCorrectionP = (float)(isPhaseOne ? 1.0F : 0.0F) *
                          solution_searchDir[workingset_nVar - 1];
    for (idx = 0; idx < totalUB; idx++) {
      totalIneq = workingset_indexUB[idx];
      pk_corrected = solution_searchDir[totalIneq - 1] - phaseOneCorrectionP;
      if ((pk_corrected > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[4] + idx) -
                                      1])) {
        alphaTemp =
            (solution_xstar[totalIneq - 1] - workingset_ub[totalIneq - 1]) -
            phaseOneCorrectionX;
        alphaTemp = fminf(fabsf(alphaTemp), tolcon - alphaTemp) / pk_corrected;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 5;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }
      }
    }
  }

  if (!isPhaseOne) {
    if ((*newBlocking) && (alpha > 1.0F)) {
      *newBlocking = false;
    }

    alpha = fminf(alpha, 1.0F);
  }

  return alpha;
}

/*
 * Arguments    : k_struct_T *obj
 *                int NColsRemain
 *                float REG_PRIMAL
 * Return Type  : void
 */
__device__ void fullColLDL2_(k_struct_T *obj, int NColsRemain,
                             float REG_PRIMAL) {
  int LDimSizeP1;
  int ijA;
  int j;
  int jA;
  int k;
  LDimSizeP1 = obj->ldm;
  for (k = 0; k < NColsRemain; k++) {
    float alpha1;
    float y;
    int LD_diagOffset;
    int b_i;
    int offset1;
    int subMatrixDim;
    LD_diagOffset = (LDimSizeP1 + 1) * k;
    if (fabsf(obj->FMat[LD_diagOffset]) <= obj->regTol_) {
      obj->FMat[LD_diagOffset] += REG_PRIMAL;
    }

    alpha1 = -1.0F / obj->FMat[LD_diagOffset];
    subMatrixDim = NColsRemain - k;
    offset1 = LD_diagOffset + 2;
    y = obj->b_workspace_;
    for (jA = 0; jA <= subMatrixDim - 2; jA++) {
      y = obj->FMat[(LD_diagOffset + jA) + 1];
    }

    obj->b_workspace_ = y;
    if (alpha1 != 0.0F) {
      jA = LD_diagOffset + LDimSizeP1;
      for (j = 0; j <= subMatrixDim - 2; j++) {
        if (y != 0.0F) {
          float temp;
          int i1;
          temp = y * alpha1;
          b_i = jA + 2;
          i1 = subMatrixDim + jA;
          for (ijA = b_i; ijA <= i1; ijA++) {
            obj->FMat[ijA - 1] += y * temp;
          }
        }

        jA += obj->ldm;
      }
    }

    alpha1 = 1.0F / obj->FMat[LD_diagOffset];
    b_i = LD_diagOffset + subMatrixDim;
    for (jA = offset1; jA <= b_i; jA++) {
      obj->FMat[jA - 1] *= alpha1;
    }
  }

  jA = (obj->ldm + 1) * (NColsRemain - 1);
  if (fabsf(obj->FMat[jA]) <= obj->regTol_) {
    obj->FMat[jA] += REG_PRIMAL;
  }
}

/*
 * Arguments    : e_struct_T *obj
 * Return Type  : void
 */
__device__ void initActiveSet(e_struct_T *obj) {
  int b_i;
  int idxFillStart;
  int idx_local;
  setProblemType(obj, 3);
  idxFillStart = obj->isActiveIdx[2];
  b_i = obj->mConstrMax;
  if (idxFillStart <= b_i) {
    (void)memset(&obj->isActiveConstr[idxFillStart + -1], 0,
                 (unsigned int)(int)((b_i - idxFillStart) + 1) * sizeof(bool));
  }

  obj->nWConstr[0] = obj->sizes[0];
  obj->nWConstr[1] = obj->sizes[1];
  obj->nWConstr[2] = 0;
  obj->nWConstr[3] = 0;
  obj->nWConstr[4] = 0;
  obj->nActiveConstr = obj->nWConstr[0] + obj->nWConstr[1];
  idxFillStart = obj->sizes[0];
  for (idx_local = 0; idx_local < idxFillStart; idx_local++) {
    int colOffsetATw;
    int i1;
    obj->Wid[idx_local] = 1;
    obj->Wlocalidx[idx_local] = idx_local + 1;
    obj->isActiveConstr[idx_local] = true;
    colOffsetATw = obj->ldA * idx_local;
    b_i = obj->indexFixed[idx_local];
    if (b_i - 2 >= 0) {
      (void)memset(
          &obj->ATwset[colOffsetATw], 0,
          (unsigned int)(int)(((b_i + colOffsetATw) - colOffsetATw) - 1) *
              sizeof(float));
    }

    obj->ATwset[(obj->indexFixed[idx_local] + colOffsetATw) - 1] = 1.0F;
    b_i = obj->indexFixed[idx_local] + 1;
    i1 = obj->nVar;
    if (b_i <= i1) {
      (void)memset(
          &obj->ATwset[(b_i + colOffsetATw) + -1], 0,
          (unsigned int)(int)((((i1 + colOffsetATw) - b_i) - colOffsetATw) +
                              1) *
              sizeof(float));
    }

    obj->bwset[idx_local] = obj->ub[obj->indexFixed[idx_local] - 1];
  }
}

/*
 * Arguments    : const float b_H[16]
 *                const float f[5]
 *                d_struct_T *solution
 *                b_struct_T *memspace
 *                e_struct_T *b_workingset
 *                c_struct_T *b_qrmanager
 *                k_struct_T *b_cholmanager
 *                j_struct_T *objective
 *                const char options_SolverName[7]
 *                float options_StepTolerance
 *                float options_ConstraintTolerance
 *                float options_ObjectiveLimit
 *                float options_PricingTolerance
 *                const g_struct_T runTimeOptions
 * Return Type  : void
 */
__device__ void iterate(
    const float b_H[16], const float f[5], d_struct_T *solution,
    b_struct_T *memspace, e_struct_T *b_workingset, c_struct_T *b_qrmanager,
    k_struct_T *b_cholmanager, j_struct_T *objective,
    const char options_SolverName[7], float options_StepTolerance,
    float options_ConstraintTolerance, float options_ObjectiveLimit,
    float options_PricingTolerance, const g_struct_T runTimeOptions) {
  static const char b[7] = {'f', 'm', 'i', 'n', 'c', 'o', 'n'};

  float b_f;
  float s;
  float tolDelta;
  int Qk0;
  int TYPE;
  int activeSetChangeID;
  int globalActiveConstrIdx;
  int idx;
  int ix0;
  int iyend;
  int k;
  int n;
  int nVar;
  bool newBlocking;
  bool subProblemChanged;
  bool updateFval;
  subProblemChanged = true;
  updateFval = true;
  activeSetChangeID = 0;
  TYPE = objective->objtype;
  tolDelta = 0.00126644492F;
  nVar = b_workingset->nVar;
  globalActiveConstrIdx = 0;
  computeGrad_StoreHx(objective, b_H, f, solution->xstar);
  solution->fstar = computeFval_ReuseHx(objective, memspace->workspace_float, f,
                                        solution->xstar);
  if (solution->iterations < runTimeOptions.MaxIterations) {
    solution->state = -5;
  } else {
    solution->state = 0;
  }

  n = b_workingset->mConstrMax;
  if (n - 1 >= 0) {
    (void)memset(&solution->lambda[0], 0, (unsigned int)n * sizeof(float));
  }

  int exitg1;
  do {
    exitg1 = 0;
    if (solution->state == -5) {
      float b_c;
      float temp;
      int b_i;
      int iy;
      bool guard1;
      bool guard2;
      guard1 = false;
      guard2 = false;
      if (subProblemChanged) {
        switch (activeSetChangeID) {
          case 1:
            ix0 = b_workingset->ldA * (b_workingset->nActiveConstr - 1);
            iyend = b_qrmanager->mrows;
            Qk0 = b_qrmanager->ncols + 1;
            if (iyend <= Qk0) {
              Qk0 = iyend;
            }

            b_qrmanager->minRowCol = Qk0;
            iy = b_qrmanager->ldq * b_qrmanager->ncols;
            Qk0 = b_qrmanager->ldq;
            if (b_qrmanager->mrows != 0) {
              iyend = iy + b_qrmanager->mrows;
              if (iy + 1 <= iyend) {
                (void)memset(&b_qrmanager->QR[iy], 0,
                             (unsigned int)(int)(iyend - iy) * sizeof(float));
              }

              b_i = b_qrmanager->ldq * (b_qrmanager->mrows - 1) + 1;
              iyend = 1;
              while (((Qk0 > 0) && (iyend <= b_i)) ||
                     ((Qk0 < 0) && (iyend >= b_i))) {
                b_c = 0.0F;
                k = (iyend + b_qrmanager->mrows) - 1;
                for (n = iyend; n <= k; n++) {
                  b_c += b_qrmanager->Q[n - 1] *
                         b_workingset->ATwset[(ix0 + n) - iyend];
                }

                b_qrmanager->QR[iy] += b_c;
                iy++;
                iyend += Qk0;
              }
            }

            b_qrmanager->ncols++;
            b_i = b_qrmanager->ncols - 1;
            b_qrmanager->jpvt[b_i] = b_qrmanager->ncols;
            idx = b_qrmanager->mrows - 2;
            while (idx + 2 > b_qrmanager->ncols) {
              k = idx + b_qrmanager->ldq * b_i;
              b_f = b_qrmanager->QR[k + 1];
              b_c = xrotg(&b_qrmanager->QR[k], &b_f, &s);
              b_qrmanager->QR[k + 1] = b_f;
              Qk0 = b_qrmanager->ldq * idx;
              n = b_qrmanager->mrows;
              if (b_qrmanager->mrows >= 1) {
                iy = b_qrmanager->ldq + Qk0;
                for (k = 0; k < n; k++) {
                  iyend = iy + k;
                  ix0 = Qk0 + k;
                  temp = b_c * b_qrmanager->Q[ix0] + s * b_qrmanager->Q[iyend];
                  b_qrmanager->Q[iyend] =
                      b_c * b_qrmanager->Q[iyend] - s * b_qrmanager->Q[ix0];
                  b_qrmanager->Q[ix0] = temp;
                }
              }

              idx--;
            }
            break;

          case -1:
            deleteColMoveEnd(b_qrmanager, globalActiveConstrIdx);
            break;

          default:
            factorQR(b_qrmanager, b_workingset->ATwset, nVar,
                     b_workingset->nActiveConstr, b_workingset->ldA);
            computeQ_(b_qrmanager, b_qrmanager->mrows);
            break;
        }

        iyend = CompareArrays(&options_SolverName[0], &b[0], 7);
        compute_deltax(b_H, solution, memspace, b_qrmanager, b_cholmanager,
                       objective, iyend == 0);
        if (solution->state != -5) {
          exitg1 = 1;
        } else if (b_xnrm2(nVar, solution->searchDir) < options_StepTolerance) {
          guard2 = true;
        } else if (b_workingset->nActiveConstr >= nVar) {
          guard2 = true;
        } else {
          updateFval = (TYPE == 5);
          if (updateFval || runTimeOptions.RemainFeasible) {
            temp = feasibleratiotest(
                solution->xstar, solution->searchDir, memspace->workspace_float,
                b_workingset->nVar, b_workingset->lb, b_workingset->ub,
                b_workingset->indexLB, b_workingset->indexUB,
                b_workingset->sizes, b_workingset->isActiveIdx,
                b_workingset->isActiveConstr, b_workingset->nWConstr,
                updateFval, options_ConstraintTolerance, &newBlocking, &iyend,
                &Qk0);
          } else {
            temp = ratiotest(
                solution->xstar, solution->searchDir, memspace->workspace_float,
                b_workingset->nVar, b_workingset->lb, b_workingset->ub,
                b_workingset->indexLB, b_workingset->indexUB,
                b_workingset->sizes, b_workingset->isActiveIdx,
                b_workingset->isActiveConstr, b_workingset->nWConstr,
                options_ConstraintTolerance, &tolDelta, &newBlocking, &iyend,
                &Qk0);
          }

          if (newBlocking) {
            switch (iyend) {
              case 3:
                addAineqConstr(b_workingset);
                break;

              case 4:
                addBoundToActiveSetMatrix_(b_workingset, 4, Qk0);
                break;

              default:
                addBoundToActiveSetMatrix_(b_workingset, 5, Qk0);
                break;
            }

            activeSetChangeID = 1;
          } else {
            if (objective->objtype == 5) {
              if (b_xnrm2(objective->b_nvar, solution->searchDir) >
                  100.0F * (float)objective->b_nvar * 0.000345266977F) {
                solution->state = 3;
              } else {
                solution->state = 4;
              }
            }

            subProblemChanged = false;
            if (b_workingset->nActiveConstr == 0) {
              solution->state = 1;
            }
          }

          if ((nVar >= 1) && (temp != 0.0F)) {
            iyend = nVar - 1;
            for (k = 0; k <= iyend; k++) {
              solution->xstar[k] += temp * solution->searchDir[k];
            }
          }

          computeGrad_StoreHx(objective, b_H, f, solution->xstar);
          updateFval = true;
          guard1 = true;
        }
      } else {
        if (nVar - 1 >= 0) {
          (void)memset(&solution->searchDir[0], 0,
                       (unsigned int)nVar * sizeof(float));
        }

        guard2 = true;
      }

      if (guard2) {
        int nActiveConstr;
        nActiveConstr = b_qrmanager->ncols;
        if (b_qrmanager->ncols > 0) {
          bool b_guard1;
          b_guard1 = false;
          if (objective->objtype != 4) {
            temp = 100.0F * (float)b_qrmanager->mrows * 1.1920929E-7F;
            if ((b_qrmanager->mrows > 0) && (b_qrmanager->ncols > 0)) {
              updateFval = true;
            } else {
              updateFval = false;
            }

            if (updateFval) {
              bool b_guard2;
              idx = b_qrmanager->ncols;
              b_guard2 = false;
              if (b_qrmanager->mrows < b_qrmanager->ncols) {
                iyend = b_qrmanager->mrows +
                        b_qrmanager->ldq * (b_qrmanager->ncols - 1);
                while ((idx > b_qrmanager->mrows) &&
                       (fabsf(b_qrmanager->QR[iyend - 1]) >= temp)) {
                  idx--;
                  iyend -= b_qrmanager->ldq;
                }

                updateFval = (idx == b_qrmanager->mrows);
                if (updateFval) {
                  b_guard2 = true;
                }
              } else {
                b_guard2 = true;
              }

              if (b_guard2) {
                iyend = idx + b_qrmanager->ldq * (idx - 1);
                while ((idx >= 1) &&
                       (fabsf(b_qrmanager->QR[iyend - 1]) >= temp)) {
                  idx--;
                  iyend = (iyend - b_qrmanager->ldq) - 1;
                }

                updateFval = (idx == 0);
              }
            }

            if (!updateFval) {
              solution->state = -7;
            } else {
              b_guard1 = true;
            }
          } else {
            b_guard1 = true;
          }

          if (b_guard1) {
            Qk0 = b_qrmanager->ldq;
            if (b_qrmanager->mrows != 0) {
              iyend = b_qrmanager->ncols;
              (void)memset(&memspace->workspace_float[0], 0,
                           (unsigned int)iyend * sizeof(float));
              iy = 0;
              b_i = b_qrmanager->ldq * (b_qrmanager->ncols - 1) + 1;
              iyend = 1;
              while (((Qk0 > 0) && (iyend <= b_i)) ||
                     ((Qk0 < 0) && (iyend >= b_i))) {
                b_c = 0.0F;
                k = (iyend + b_qrmanager->mrows) - 1;
                for (n = iyend; n <= k; n++) {
                  b_c += b_qrmanager->Q[n - 1] * objective->grad[n - iyend];
                }

                memspace->workspace_float[iy] += b_c;
                iy++;
                iyend += Qk0;
              }
            }

            n = b_qrmanager->ncols;
            if (b_qrmanager->ncols != 0) {
              for (ix0 = n; ix0 >= 1; ix0--) {
                Qk0 = (ix0 + (ix0 - 1) * b_qrmanager->ldq) - 1;
                memspace->workspace_float[ix0 - 1] /= b_qrmanager->QR[Qk0];
                for (k = 0; k <= ix0 - 2; k++) {
                  iyend = (ix0 - k) - 2;
                  memspace->workspace_float[iyend] -=
                      memspace->workspace_float[ix0 - 1] *
                      b_qrmanager->QR[(Qk0 - k) - 1];
                }
              }
            }

            for (idx = 0; idx < nActiveConstr; idx++) {
              solution->lambda[idx] = -memspace->workspace_float[idx];
            }
          }
        }

        if ((solution->state != -7) || (b_workingset->nActiveConstr > nVar)) {
          iyend = 0;
          temp = options_PricingTolerance * runTimeOptions.ProbRelTolFactor *
                 (float)(TYPE != 5 ? (int)1 : (int)0);
          b_i = (b_workingset->nWConstr[0] + b_workingset->nWConstr[1]) + 1;
          k = b_workingset->nActiveConstr;
          for (idx = b_i; idx <= k; idx++) {
            b_f = solution->lambda[idx - 1];
            if (b_f < temp) {
              temp = b_f;
              iyend = idx;
            }
          }

          if (iyend == 0) {
            solution->state = 1;
          } else {
            activeSetChangeID = -1;
            globalActiveConstrIdx = iyend;
            subProblemChanged = true;
            removeConstr(b_workingset, iyend);
            if (iyend < b_workingset->nActiveConstr + 1) {
              solution->lambda[iyend - 1] =
                  solution->lambda[b_workingset->nActiveConstr];
            }

            solution->lambda[b_workingset->nActiveConstr] = 0.0F;
          }
        } else {
          iyend = b_workingset->nActiveConstr;
          activeSetChangeID = 0;
          globalActiveConstrIdx = b_workingset->nActiveConstr;
          subProblemChanged = true;
          removeConstr(b_workingset, b_workingset->nActiveConstr);
          solution->lambda[iyend - 1] = 0.0F;
        }

        updateFval = false;
        guard1 = true;
      }

      if (guard1) {
        solution->iterations++;
        iyend = objective->b_nvar - 1;
        if ((solution->iterations >= runTimeOptions.MaxIterations) &&
            ((solution->state != 1) || (objective->objtype == 5))) {
          solution->state = 0;
        }

        if (solution->iterations - solution->iterations / 50 * 50 == 0) {
          solution->maxConstr =
              b_maxConstraintViolation(b_workingset, solution->xstar);
          temp = solution->maxConstr;
          if (objective->objtype == 5) {
            temp = solution->maxConstr - solution->xstar[iyend];
          }

          if (temp >
              options_ConstraintTolerance * runTimeOptions.ConstrRelTolFactor) {
            if (iyend >= 0) {
              (void)memcpy(&solution->searchDir[0], &solution->xstar[0],
                           (unsigned int)(int)(iyend + 1) * sizeof(float));
            }

            newBlocking = feasibleX0ForWorkingSet(memspace->workspace_float,
                                                  solution->searchDir,
                                                  b_workingset, b_qrmanager);
            if ((!newBlocking) && (solution->state != 0)) {
              solution->state = -2;
            }

            activeSetChangeID = 0;
            temp = b_maxConstraintViolation(b_workingset, solution->searchDir);
            if (temp < solution->maxConstr) {
              if (iyend >= 0) {
                (void)memcpy(&solution->xstar[0], &solution->searchDir[0],
                             (unsigned int)(int)(iyend + 1) * sizeof(float));
              }

              solution->maxConstr = temp;
            }
          }
        }

        if (updateFval) {
          solution->fstar = computeFval_ReuseHx(
              objective, memspace->workspace_float, f, solution->xstar);
          if ((solution->fstar < options_ObjectiveLimit) &&
              ((solution->state != 0) || (objective->objtype != 5))) {
            solution->state = 2;
          }
        }
      }
    } else {
      if (!updateFval) {
        solution->fstar = computeFval_ReuseHx(
            objective, memspace->workspace_float, f, solution->xstar);
      }

      exitg1 = 1;
    }
  } while (exitg1 == 0);
}

/*
 * Arguments    : int n
 *                const float x[81]
 *                int incx
 * Return Type  : int
 */
__device__ int ixamax(int n, const float x[81], int incx) {
  int idxmax;
  int k;
  if ((n < 1) || (incx < 1)) {
    idxmax = 0;
  } else {
    idxmax = 1;
    if (n > 1) {
      float smax;
      smax = fabsf(x[0]);
      for (k = 2; k <= n; k++) {
        float s;
        s = fabsf(x[(k - 1) * incx]);
        if (s > smax) {
          idxmax = k;
          smax = s;
        }
      }
    }
  }

  return idxmax;
}

/*
 * Arguments    : bool obj_hasLinear
 *                int obj_nvar
 *                float workspace[45]
 *                const float b_H[16]
 *                const float f[5]
 *                const float x[5]
 * Return Type  : void
 */
__device__ void linearForm_(bool obj_hasLinear, int obj_nvar,
                            float workspace[45], const float b_H[16],
                            const float f[5], const float x[5]) {
  int ia;
  int ix;
  ix = 0;
  if (obj_hasLinear) {
    if (obj_nvar - 1 >= 0) {
      (void)memcpy(&workspace[0], &f[0],
                   (unsigned int)obj_nvar * sizeof(float));
    }

    ix = 1;
  }

  if (obj_nvar != 0) {
    int b_i;
    int iac;
    if ((ix != 1) && (obj_nvar - 1 >= 0)) {
      (void)memset(&workspace[0], 0, (unsigned int)obj_nvar * sizeof(float));
    }

    ix = 0;
    b_i = obj_nvar * (obj_nvar - 1) + 1;
    iac = 1;
    while (((obj_nvar > 0) && (iac <= b_i)) ||
           ((obj_nvar < 0) && (iac >= b_i))) {
      float b_c;
      int i1;
      b_c = 0.5F * x[ix];
      i1 = (iac + obj_nvar) - 1;
      for (ia = iac; ia <= i1; ia++) {
        int i2;
        i2 = ia - iac;
        workspace[i2] += b_H[ia - 1] * b_c;
      }

      ix++;
      iac += obj_nvar;
    }
  }
}

/*
 * Arguments    : const e_struct_T *obj
 *                const float x[45]
 *                int ix0
 * Return Type  : float
 */
__device__ float maxConstraintViolation(const e_struct_T *obj,
                                        const float x[45], int ix0) {
  float v;
  int idx;
  int mFixed;
  int mLB;
  int mUB;
  mLB = obj->sizes[3];
  mUB = obj->sizes[4];
  mFixed = obj->sizes[0];
  v = 0.0F;
  if (obj->sizes[3] > 0) {
    for (idx = 0; idx < mLB; idx++) {
      v = fmaxf(v, -x[(ix0 + obj->indexLB[idx]) - 2] -
                       obj->lb[obj->indexLB[idx] - 1]);
    }
  }

  if (obj->sizes[4] > 0) {
    for (idx = 0; idx < mUB; idx++) {
      v = fmaxf(
          v, x[(ix0 + obj->indexUB[idx]) - 2] - obj->ub[obj->indexUB[idx] - 1]);
    }
  }

  if (obj->sizes[0] > 0) {
    for (idx = 0; idx < mFixed; idx++) {
      v = fmaxf(v, fabsf(x[(ix0 + obj->indexFixed[idx]) - 2] -
                         obj->ub[obj->indexFixed[idx] - 1]));
    }
  }

  return v;
}

/*
 * Arguments    : e_struct_T *obj
 * Return Type  : void
 */
__device__ void modifyOverheadPhaseOne_(e_struct_T *obj) {
  int b_i;
  int idx;
  int idxStartIneq;
  b_i = obj->sizes[0];
  for (idx = 0; idx < b_i; idx++) {
    obj->ATwset[(obj->nVar + obj->ldA * idx) - 1] = 0.0F;
  }

  obj->indexLB[obj->sizes[3] - 1] = obj->nVar;
  obj->lb[obj->nVar - 1] = (float)obj->SLACK0;
  idxStartIneq = obj->isActiveIdx[2];
  b_i = obj->nActiveConstr;
  for (idx = idxStartIneq; idx <= b_i; idx++) {
    obj->ATwset[(obj->nVar + obj->ldA * (idx - 1)) - 1] = -1.0F;
  }

  idxStartIneq = obj->isActiveIdx[4] - 1;
  if (obj->nWConstr[4] > 0) {
    b_i = obj->sizesNormal[4] - 1;
    for (idx = b_i; idx >= 0; idx--) {
      int i1;
      i1 = idxStartIneq + idx;
      obj->isActiveConstr[i1] = obj->isActiveConstr[i1 - 1];
    }
  } else {
    obj->isActiveConstr[(obj->isActiveIdx[4] + obj->sizesNormal[4]) - 1] =
        false;
  }

  obj->isActiveConstr[obj->isActiveIdx[4] - 2] = false;
}

/*
 * Arguments    : float b_A[81]
 *                int m
 *                int n
 *                int nfxd
 *                float tau[9]
 * Return Type  : void
 */
__device__ void qrf(float b_A[81], int m, int n, int nfxd, float tau[9]) {
  float work[9];
  float atmp;
  int b_i;
  for (b_i = 0; b_i < 9; b_i++) {
    tau[b_i] = 0.0F;
    work[b_i] = 0.0F;
  }

  for (b_i = 0; b_i < nfxd; b_i++) {
    float f;
    int b_ii;
    int mmi;
    b_ii = b_i * 9 + b_i;
    mmi = m - b_i;
    if (b_i + 1 < m) {
      atmp = b_A[b_ii];
      f = xzlarfg(mmi, &atmp, b_A, b_ii + 2);
      tau[b_i] = f;
      b_A[b_ii] = atmp;
    } else {
      f = 0.0F;
      tau[b_i] = 0.0F;
    }

    if (b_i + 1 < n) {
      atmp = b_A[b_ii];
      b_A[b_ii] = 1.0F;
      xzlarf(mmi, (n - b_i) - 1, b_ii + 1, f, b_A, b_ii + 10, work);
      b_A[b_ii] = atmp;
    }
  }
}

/*
 * Arguments    : const float solution_xstar[5]
 *                const float solution_searchDir[5]
 *                const float workspace[45]
 *                int workingset_nVar
 *                const float workingset_lb[5]
 *                const float workingset_ub[5]
 *                const int workingset_indexLB[5]
 *                const int workingset_indexUB[5]
 *                const int b_workingset_sizes[5]
 *                const int b_workingset_isActiveIdx[6]
 *                const bool workingset_isActiveConstr[9]
 *                const int workingset_nWConstr[5]
 *                float tolcon
 *                float *b_toldelta
 *                bool *newBlocking
 *                int *constrType
 *                int *constrIdx
 * Return Type  : float
 */
__device__ float ratiotest(
    const float solution_xstar[5], const float solution_searchDir[5],
    const float workspace[45], int workingset_nVar,
    const float workingset_lb[5], const float workingset_ub[5],
    const int workingset_indexLB[5], const int workingset_indexUB[5],
    const int b_workingset_sizes[5], const int b_workingset_isActiveIdx[6],
    const bool workingset_isActiveConstr[9], const int workingset_nWConstr[5],
    float tolcon, float *b_toldelta, bool *newBlocking, int *constrType,
    int *constrIdx) {
  float alpha;
  float alphaTemp;
  float denomTol;
  float f;
  float p_max;
  float pk_corrected;
  float ratio;
  float ratio_tmp;
  int idx;
  int totalIneq;
  int totalUB;
  totalIneq = b_workingset_sizes[2];
  totalUB = b_workingset_sizes[4];
  alpha = 1.0E+30F;
  *newBlocking = false;
  *constrType = 0;
  *constrIdx = 0;
  p_max = 0.0F;
  denomTol = 0.00011920929F * b_xnrm2(workingset_nVar, solution_searchDir);
  if (workingset_nWConstr[2] < b_workingset_sizes[2]) {
    for (idx = 0; idx < totalIneq; idx++) {
      f = workspace[idx + 9];
      if ((f > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[2] + idx) -
                                      1])) {
        pk_corrected = workspace[idx];
        ratio = tolcon - pk_corrected;
        alphaTemp =
            fminf(fabsf(pk_corrected - *b_toldelta), ratio + *b_toldelta) / f;
        if ((alphaTemp <= alpha) && (fabsf(f) > p_max)) {
          alpha = alphaTemp;
          *constrType = 3;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }

        alphaTemp = fminf(fabsf(pk_corrected), ratio) / f;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 3;
          *constrIdx = idx + 1;
          *newBlocking = true;
          p_max = fabsf(f);
        }
      }
    }
  }

  if (workingset_nWConstr[3] < b_workingset_sizes[3]) {
    totalIneq = b_workingset_sizes[3];
    for (idx = 0; idx <= totalIneq - 2; idx++) {
      int b_i;
      b_i = workingset_indexLB[idx];
      pk_corrected = -solution_searchDir[b_i - 1];
      if ((pk_corrected > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[3] + idx) -
                                      1])) {
        ratio_tmp = -solution_xstar[b_i - 1] - workingset_lb[b_i - 1];
        ratio = ratio_tmp - *b_toldelta;
        alphaTemp = fminf(fabsf(ratio), tolcon - ratio) / pk_corrected;
        if ((alphaTemp <= alpha) && (fabsf(pk_corrected) > p_max)) {
          alpha = alphaTemp;
          *constrType = 4;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }

        alphaTemp = fminf(fabsf(ratio_tmp), tolcon - ratio_tmp) / pk_corrected;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 4;
          *constrIdx = idx + 1;
          *newBlocking = true;
          p_max = fabsf(pk_corrected);
        }
      }
    }

    totalIneq = workingset_indexLB[b_workingset_sizes[3] - 1] - 1;
    f = solution_searchDir[totalIneq];
    if ((-f > denomTol) &&
        (!workingset_isActiveConstr
             [(b_workingset_isActiveIdx[3] + b_workingset_sizes[3]) - 2])) {
      ratio_tmp = -solution_xstar[totalIneq] - workingset_lb[totalIneq];
      ratio = ratio_tmp - *b_toldelta;
      alphaTemp = fminf(fabsf(ratio), tolcon - ratio) / -f;
      if ((alphaTemp <= alpha) && (fabsf(f) > p_max)) {
        alpha = alphaTemp;
        *constrType = 4;
        *constrIdx = b_workingset_sizes[3];
        *newBlocking = true;
      }

      alphaTemp = fminf(fabsf(ratio_tmp), tolcon - ratio_tmp) / -f;
      if (alphaTemp < alpha) {
        alpha = alphaTemp;
        *constrType = 4;
        *constrIdx = b_workingset_sizes[3];
        *newBlocking = true;
        p_max = fabsf(f);
      }
    }
  }

  if (workingset_nWConstr[4] < b_workingset_sizes[4]) {
    for (idx = 0; idx < totalUB; idx++) {
      totalIneq = workingset_indexUB[idx];
      pk_corrected = solution_searchDir[totalIneq - 1];
      if ((pk_corrected > denomTol) &&
          (!workingset_isActiveConstr[(b_workingset_isActiveIdx[4] + idx) -
                                      1])) {
        ratio_tmp =
            solution_xstar[totalIneq - 1] - workingset_ub[totalIneq - 1];
        ratio = ratio_tmp - *b_toldelta;
        alphaTemp = fminf(fabsf(ratio), tolcon - ratio) / pk_corrected;
        if ((alphaTemp <= alpha) && (fabsf(pk_corrected) > p_max)) {
          alpha = alphaTemp;
          *constrType = 5;
          *constrIdx = idx + 1;
          *newBlocking = true;
        }

        alphaTemp = fminf(fabsf(ratio_tmp), tolcon - ratio_tmp) / pk_corrected;
        if (alphaTemp < alpha) {
          alpha = alphaTemp;
          *constrType = 5;
          *constrIdx = idx + 1;
          *newBlocking = true;
          p_max = fabsf(pk_corrected);
        }
      }
    }
  }

  *b_toldelta += 0.00124111609F;
  if (p_max > 0.0F) {
    alpha = fmaxf(alpha, 0.00124111609F / p_max);
  }

  if ((*newBlocking) && (alpha > 1.0F)) {
    *newBlocking = false;
  }

  return fminf(alpha, 1.0F);
}

/*
 * Arguments    : const float Hessian[16]
 *                const float grad[5]
 *                d_struct_T *TrialState
 *                i_struct_T *MeritFunction
 *                b_struct_T *memspace
 *                e_struct_T *WorkingSet
 *                c_struct_T *QRManager
 *                k_struct_T *CholManager
 *                j_struct_T *QPObjective
 *                g_struct_T *qpoptions
 * Return Type  : void
 */
__device__ void relaxed(const float Hessian[16], const float grad[5],
                        d_struct_T *TrialState, i_struct_T *MeritFunction,
                        b_struct_T *memspace, e_struct_T *WorkingSet,
                        c_struct_T *QRManager, k_struct_T *CholManager,
                        j_struct_T *QPObjective, g_struct_T *qpoptions) {
  float beta;
  float s;
  float smax;
  int b_idx_max;
  int b_mEq;
  int idx;
  int idx_negative;
  int mEq;
  int mIneq;
  int nActiveLBArtificial;
  int nArtificial_tmp;
  int nVarOrig;
  int temp;
  bool tf;
  nVarOrig = WorkingSet->nVar - 1;
  mEq = WorkingSet->sizes[1];
  beta = 0.0F;
  idx = 0;
  for (temp = 0; temp <= nVarOrig; temp++) {
    beta += Hessian[idx];
    idx += 5;
  }

  beta /= (float)WorkingSet->nVar;
  if (TrialState->sqpIterations <= 1) {
    temp = QPObjective->b_nvar;
    if (QPObjective->b_nvar < 1) {
      b_idx_max = 0;
    } else {
      b_idx_max = 1;
      if (QPObjective->b_nvar > 1) {
        smax = fabsf(grad[0]);
        for (idx_negative = 2; idx_negative <= temp; idx_negative++) {
          s = fabsf(grad[idx_negative - 1]);
          if (s > smax) {
            b_idx_max = idx_negative;
            smax = s;
          }
        }
      }
    }

    smax = 100.0F * fmaxf(1.0F, fabsf(grad[b_idx_max - 1]));
  } else {
    temp = WorkingSet->mConstr;
    if (WorkingSet->mConstr < 1) {
      b_idx_max = 0;
    } else {
      b_idx_max = 1;
      if (WorkingSet->mConstr > 1) {
        smax = fabsf(TrialState->lambdasqp[0]);
        for (idx_negative = 2; idx_negative <= temp; idx_negative++) {
          s = fabsf(TrialState->lambdasqp[idx_negative - 1]);
          if (s > smax) {
            b_idx_max = idx_negative;
            smax = s;
          }
        }
      }
    }

    smax = fabsf(TrialState->lambdasqp[b_idx_max - 1]);
  }

  QPObjective->b_nvar = WorkingSet->nVar;
  QPObjective->beta = beta;
  QPObjective->rho = smax;
  QPObjective->hasLinear = true;
  QPObjective->objtype = 4;
  setProblemType(WorkingSet, 2);
  mIneq = WorkingSet->sizes[2] + 1;
  b_mEq = WorkingSet->sizes[1];
  temp =
      (WorkingSet->sizes[3] - 2 * WorkingSet->sizes[1]) - WorkingSet->sizes[2];
  for (idx = 0; idx <= mIneq - 2; idx++) {
    s = memspace->workspace_float[idx];
    TrialState->xstar[(nVarOrig + idx) + 1] =
        (float)(s > 0.0F ? (int)1 : (int)0) * s;
  }

  for (idx = 0; idx < b_mEq; idx++) {
    b_idx_max = mIneq + idx;
    idx_negative = (mIneq + b_mEq) + idx;
    s = memspace->workspace_float[idx];
    if (s <= 0.0F) {
      TrialState->xstar[nVarOrig + b_idx_max] = 0.0F;
      TrialState->xstar[nVarOrig + idx_negative] = -s;
      addBoundToActiveSetMatrix_(WorkingSet, 4, temp + b_idx_max);
      if (s >= -0.001F) {
        addBoundToActiveSetMatrix_(WorkingSet, 4, temp + idx_negative);
      }
    } else {
      TrialState->xstar[nVarOrig + b_idx_max] = s;
      TrialState->xstar[nVarOrig + idx_negative] = 0.0F;
      addBoundToActiveSetMatrix_(WorkingSet, 4, temp + idx_negative);
      if (s <= 0.001F) {
        addBoundToActiveSetMatrix_(WorkingSet, 4, temp + b_idx_max);
      }
    }
  }

  temp = qpoptions->MaxIterations;
  qpoptions->MaxIterations =
      ((qpoptions->MaxIterations + WorkingSet->nVar) - nVarOrig) - 1;
  driver(Hessian, grad, TrialState, memspace, WorkingSet, QRManager,
         CholManager, QPObjective, *qpoptions, *qpoptions);
  qpoptions->MaxIterations = temp;
  mIneq = WorkingSet->sizes[2];
  b_mEq = WorkingSet->sizes[1];
  nArtificial_tmp = 2 * WorkingSet->sizes[1];
  temp = nArtificial_tmp + WorkingSet->sizes[2];
  b_idx_max = WorkingSet->sizes[3] - 1;
  nActiveLBArtificial = 0;
  for (idx = 0; idx < b_mEq; idx++) {
    bool b_tf;
    idx_negative = WorkingSet->isActiveIdx[3] + b_idx_max;
    tf = WorkingSet->isActiveConstr[(idx_negative - 2 * b_mEq) + idx];
    b_tf = WorkingSet->isActiveConstr[(idx_negative - b_mEq) + idx];
    memspace->workspace_int[idx] = tf ? (int)1 : (int)0;
    memspace->workspace_int[idx + b_mEq] = b_tf ? (int)1 : (int)0;
    nActiveLBArtificial = (nActiveLBArtificial + (tf ? (int)1 : (int)0)) +
                          (b_tf ? (int)1 : (int)0);
  }

  for (idx = 0; idx < mIneq; idx++) {
    tf =
        WorkingSet
            ->isActiveConstr[((WorkingSet->isActiveIdx[3] + b_idx_max) - temp) +
                             idx];
    memspace->workspace_int[idx + nArtificial_tmp] = tf ? (int)1 : (int)0;
    nActiveLBArtificial += tf ? (int)1 : (int)0;
  }

  if (TrialState->state != -6) {
    float qpfvalLinearExcess;
    float qpfvalQuadExcess;
    nArtificial_tmp = (WorkingSet->nVarMax - nVarOrig) - 2;
    temp = nVarOrig + 2;
    qpfvalLinearExcess = 0.0F;
    qpfvalQuadExcess = 0.0F;
    if (nArtificial_tmp >= 1) {
      b_idx_max = nVarOrig + nArtificial_tmp;
      for (idx_negative = temp; idx_negative <= b_idx_max + 1; idx_negative++) {
        qpfvalLinearExcess += fabsf(TrialState->xstar[idx_negative - 1]);
      }
    }

    if (nArtificial_tmp >= 1) {
      for (idx_negative = 0; idx_negative < nArtificial_tmp; idx_negative++) {
        s = TrialState->xstar[(nVarOrig + idx_negative) + 1];
        qpfvalQuadExcess += s * s;
      }
    }

    qpfvalLinearExcess = (TrialState->fstar - smax * qpfvalLinearExcess) -
                         beta / 2.0F * qpfvalQuadExcess;
    qpfvalQuadExcess = MeritFunction->penaltyParam;
    smax = MeritFunction->linearizedConstrViol;
    s = 0.0F;
    if (nArtificial_tmp >= 1) {
      b_idx_max = nVarOrig + nArtificial_tmp;
      for (idx_negative = temp; idx_negative <= b_idx_max + 1; idx_negative++) {
        s += fabsf(TrialState->xstar[idx_negative - 1]);
      }
    }

    MeritFunction->linearizedConstrViol = s;
    smax -= s;
    if ((smax > 1.1920929E-7F) && (qpfvalLinearExcess > 0.0F)) {
      if (TrialState->sqpFval == 0.0F) {
        beta = 1.0F;
      } else {
        beta = 1.5F;
      }

      qpfvalQuadExcess = beta * qpfvalLinearExcess / smax;
    }

    if (qpfvalQuadExcess < MeritFunction->penaltyParam) {
      MeritFunction->phi = TrialState->sqpFval;
      if ((MeritFunction->initFval +
           qpfvalQuadExcess * (MeritFunction->initConstrViolationEq +
                               MeritFunction->initConstrViolationIneq)) -
              TrialState->sqpFval >
          (float)MeritFunction->nPenaltyDecreases * MeritFunction->threshold) {
        MeritFunction->nPenaltyDecreases++;
        if (MeritFunction->nPenaltyDecreases * 2 > TrialState->sqpIterations) {
          MeritFunction->threshold *= 10.0F;
        }

        MeritFunction->penaltyParam = fmaxf(qpfvalQuadExcess, 1.1920929E-7F);
      } else {
        MeritFunction->phi = TrialState->sqpFval;
      }
    } else {
      MeritFunction->penaltyParam = fmaxf(qpfvalQuadExcess, 1.1920929E-7F);
      MeritFunction->phi = TrialState->sqpFval;
    }

    MeritFunction->phiPrimePlus = fminf(qpfvalLinearExcess, 0.0F);
    temp = WorkingSet->isActiveIdx[1] - 2;
    for (idx = 0; idx < mEq; idx++) {
      if ((memspace->workspace_int[idx] != 0) &&
          (memspace->workspace_int[idx + mEq] != 0)) {
        tf = true;
      } else {
        tf = false;
      }

      idx_negative = (temp + idx) + 1;
      TrialState->lambda[idx_negative] *= (float)(tf ? 1.0F : 0.0F);
    }

    temp = WorkingSet->isActiveIdx[2];
    b_idx_max = WorkingSet->nActiveConstr;
    for (idx = temp; idx <= b_idx_max; idx++) {
      if (WorkingSet->Wid[idx - 1] == 3) {
        TrialState->lambda[idx - 1] *=
            (float)memspace
                ->workspace_int[(WorkingSet->Wlocalidx[idx - 1] + 2 * mEq) - 1];
      }
    }
  }

  temp = WorkingSet->sizes[0];
  mEq = WorkingSet->sizes[1];
  idx_negative =
      (WorkingSet->sizes[3] - 2 * WorkingSet->sizes[1]) - WorkingSet->sizes[2];
  idx = WorkingSet->nActiveConstr;
  while ((idx > temp + mEq) && (nActiveLBArtificial > 0)) {
    if ((WorkingSet->Wid[idx - 1] == 4) &&
        (WorkingSet->Wlocalidx[idx - 1] > idx_negative)) {
      b_idx_max = WorkingSet->nActiveConstr - 1;
      smax = TrialState->lambda[b_idx_max];
      TrialState->lambda[b_idx_max] = 0.0F;
      TrialState->lambda[idx - 1] = smax;
      removeConstr(WorkingSet, idx);
      nActiveLBArtificial--;
    }

    idx--;
  }

  QPObjective->b_nvar = nVarOrig + 1;
  QPObjective->hasLinear = true;
  QPObjective->objtype = 3;
  setProblemType(WorkingSet, 3);
  sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr, WorkingSet->sizes,
               WorkingSet->isActiveIdx, WorkingSet->Wid, WorkingSet->Wlocalidx,
               memspace->workspace_float);
}

/*
 * Arguments    : e_struct_T *obj
 *                int idx_global
 * Return Type  : void
 */
__device__ void removeConstr(e_struct_T *obj, int idx_global) {
  int TYPE_tmp;
  int idx;
  TYPE_tmp = obj->Wid[idx_global - 1] - 1;
  obj->isActiveConstr[(obj->isActiveIdx[TYPE_tmp] +
                       obj->Wlocalidx[idx_global - 1]) -
                      2] = false;
  if (idx_global < obj->nActiveConstr) {
    int b_i;
    int i1;
    b_i = obj->nActiveConstr - 1;
    obj->Wid[idx_global - 1] = obj->Wid[b_i];
    obj->Wlocalidx[idx_global - 1] = obj->Wlocalidx[b_i];
    i1 = obj->nVar;
    for (idx = 0; idx < i1; idx++) {
      obj->ATwset[idx + obj->ldA * (idx_global - 1)] =
          obj->ATwset[idx + obj->ldA * b_i];
    }

    obj->bwset[idx_global - 1] = obj->bwset[b_i];
  }

  obj->nActiveConstr--;
  obj->nWConstr[TYPE_tmp]--;
}

/*
 * Arguments    : float u0
 *                float u1
 * Return Type  : float
 */
__device__ float rt_hypotf(float u0, float u1) {
  float a;
  float b;
  float y;
  a = fabsf(u0);
  b = fabsf(u1);
  if (a < b) {
    a /= b;
    y = b * sqrtf(a * a + 1.0F);
  } else if (a > b) {
    b /= a;
    y = a * sqrtf(b * b + 1.0F);
  } else {
    y = a * 1.41421354F;
  }

  return y;
}

/*
 * Arguments    : e_struct_T *obj
 *                int PROBLEM_TYPE
 * Return Type  : void
 */
__device__ void setProblemType(e_struct_T *obj, int PROBLEM_TYPE) {
  int b_i;
  int idxStartIneq;
  int idx_col;
  switch (PROBLEM_TYPE) {
    case 3: {
      obj->nVar = obj->nVarOrig;
      obj->mConstr = obj->mConstrOrig;
      if (obj->nWConstr[4] > 0) {
        int idxUpperExisting;
        idxUpperExisting = obj->isActiveIdx[4] - 2;
        b_i = obj->sizesNormal[4];
        for (idxStartIneq = 0; idxStartIneq < b_i; idxStartIneq++) {
          int i1;
          i1 = (idxUpperExisting + idxStartIneq) + 1;
          obj->isActiveConstr[(obj->isActiveIdxNormal[4] + idxStartIneq) - 1] =
              obj->isActiveConstr[i1];
          obj->isActiveConstr[i1] = false;
        }
      }

      for (b_i = 0; b_i < 5; b_i++) {
        obj->sizes[b_i] = obj->sizesNormal[b_i];
      }

      for (b_i = 0; b_i < 6; b_i++) {
        obj->isActiveIdx[b_i] = obj->isActiveIdxNormal[b_i];
      }
    } break;

    case 1:
      obj->nVar = obj->nVarOrig + 1;
      obj->mConstr = obj->mConstrOrig + 1;
      for (b_i = 0; b_i < 5; b_i++) {
        obj->sizes[b_i] = obj->sizesPhaseOne[b_i];
      }

      modifyOverheadPhaseOne_(obj);
      for (b_i = 0; b_i < 6; b_i++) {
        obj->isActiveIdx[b_i] = obj->isActiveIdxPhaseOne[b_i];
      }
      break;

    case 2: {
      obj->nVar = obj->nVarMax - 1;
      obj->mConstr = obj->mConstrMax - 1;
      for (b_i = 0; b_i < 5; b_i++) {
        obj->sizes[b_i] = obj->sizesRegularized[b_i];
      }

      if (obj->probType != 4) {
        int i1;
        int idxUpperExisting;
        int offsetIneq_tmp;
        offsetIneq_tmp = obj->nVarOrig + 1;
        b_i = obj->sizes[0];
        for (idx_col = 0; idx_col < b_i; idx_col++) {
          idxUpperExisting = obj->ldA * idx_col;
          i1 = obj->nVar;
          if (offsetIneq_tmp <= i1) {
            (void)memset(&obj->ATwset[(offsetIneq_tmp + idxUpperExisting) + -1],
                         0,
                         (unsigned int)(int)((((i1 + idxUpperExisting) -
                                               offsetIneq_tmp) -
                                              idxUpperExisting) +
                                             1) *
                             sizeof(float));
          }
        }

        idxUpperExisting = obj->nVarOrig;
        b_i = obj->sizesNormal[3] + 1;
        i1 = obj->sizesRegularized[3];
        for (idxStartIneq = b_i; idxStartIneq <= i1; idxStartIneq++) {
          idxUpperExisting++;
          obj->indexLB[idxStartIneq - 1] = idxUpperExisting;
        }

        if (obj->nWConstr[4] > 0) {
          b_i = obj->sizesRegularized[4];
          for (idxStartIneq = 0; idxStartIneq < b_i; idxStartIneq++) {
            obj->isActiveConstr[obj->isActiveIdxRegularized[4] + idxStartIneq] =
                obj->isActiveConstr[(obj->isActiveIdx[4] + idxStartIneq) - 1];
          }
        }

        b_i = obj->isActiveIdx[4];
        i1 = obj->isActiveIdxRegularized[4] - 1;
        if (b_i <= i1) {
          (void)memset(&obj->isActiveConstr[b_i + -1], 0,
                       (unsigned int)(int)((i1 - b_i) + 1) * sizeof(bool));
        }

        b_i = (obj->nVarOrig + obj->sizes[2]) + 2 * obj->sizes[1];
        if (offsetIneq_tmp <= b_i) {
          (void)memset(
              &obj->lb[offsetIneq_tmp + -1], 0,
              (unsigned int)(int)((b_i - offsetIneq_tmp) + 1) * sizeof(float));
        }

        idxStartIneq = obj->isActiveIdx[2];
        b_i = obj->nActiveConstr;
        for (idx_col = idxStartIneq; idx_col <= b_i; idx_col++) {
          idxUpperExisting = obj->ldA * (idx_col - 1) - 1;
          if (obj->Wid[idx_col - 1] == 3) {
            int i2;
            i1 = offsetIneq_tmp + obj->Wlocalidx[idx_col - 1];
            i2 = i1 - 2;
            if (offsetIneq_tmp <= i2) {
              (void)memset(&obj->ATwset[offsetIneq_tmp + idxUpperExisting], 0,
                           (unsigned int)(int)((((i2 + idxUpperExisting) -
                                                 offsetIneq_tmp) -
                                                idxUpperExisting) +
                                               1) *
                               sizeof(float));
            }

            obj->ATwset[(i1 + idxUpperExisting) - 1] = -1.0F;
            i2 = obj->nVar;
            if (i1 <= i2) {
              (void)memset(&obj->ATwset[i1 + idxUpperExisting], 0,
                           (unsigned int)(int)((((i2 + idxUpperExisting) - i1) -
                                                idxUpperExisting) +
                                               1) *
                               sizeof(float));
            }
          } else {
            i1 = obj->nVar;
            if (offsetIneq_tmp <= i1) {
              (void)memset(&obj->ATwset[offsetIneq_tmp + idxUpperExisting], 0,
                           (unsigned int)(int)((((i1 + idxUpperExisting) -
                                                 offsetIneq_tmp) -
                                                idxUpperExisting) +
                                               1) *
                               sizeof(float));
            }
          }
        }
      }

      for (b_i = 0; b_i < 6; b_i++) {
        obj->isActiveIdx[b_i] = obj->isActiveIdxRegularized[b_i];
      }
    } break;

    default:
      obj->nVar = obj->nVarMax;
      obj->mConstr = obj->mConstrMax;
      for (b_i = 0; b_i < 5; b_i++) {
        obj->sizes[b_i] = obj->sizesRegPhaseOne[b_i];
      }

      modifyOverheadPhaseOne_(obj);
      for (b_i = 0; b_i < 6; b_i++) {
        obj->isActiveIdx[b_i] = obj->isActiveIdxRegPhaseOne[b_i];
      }
      break;
  }

  obj->probType = PROBLEM_TYPE;
}

/*
 * Arguments    : const k_struct_T *obj
 *                float rhs[5]
 * Return Type  : void
 */
__device__ void solve(const k_struct_T *obj, float rhs[5]) {
  int b_i;
  int j;
  int jA;
  int n_tmp;
  n_tmp = obj->ndims;
  if (obj->ndims != 0) {
    for (j = 0; j < n_tmp; j++) {
      float temp;
      jA = j * obj->ldm;
      temp = rhs[j];
      for (b_i = 0; b_i < j; b_i++) {
        temp -= obj->FMat[jA + b_i] * rhs[b_i];
      }

      rhs[j] = temp / obj->FMat[jA + j];
    }
  }

  if (obj->ndims != 0) {
    for (j = n_tmp; j >= 1; j--) {
      jA = (j + (j - 1) * obj->ldm) - 1;
      rhs[j - 1] /= obj->FMat[jA];
      for (b_i = 0; b_i <= j - 2; b_i++) {
        int ix;
        ix = (j - b_i) - 2;
        rhs[ix] -= rhs[j - 1] * obj->FMat[(jA - b_i) - 1];
      }
    }
  }
}

/*
 * Arguments    : float lambda[9]
 *                int WorkingSet_nActiveConstr
 *                const int WorkingSet_sizes[5]
 *                const int WorkingSet_isActiveIdx[6]
 *                const int WorkingSet_Wid[9]
 *                const int WorkingSet_Wlocalidx[9]
 *                float workspace[45]
 * Return Type  : void
 */
__device__ void sortLambdaQP(float lambda[9], int WorkingSet_nActiveConstr,
                             const int WorkingSet_sizes[5],
                             const int WorkingSet_isActiveIdx[6],
                             const int WorkingSet_Wid[9],
                             const int WorkingSet_Wlocalidx[9],
                             float workspace[45]) {
  int idx;
  if (WorkingSet_nActiveConstr != 0) {
    int idxOffset;
    int mAll;
    mAll =
        ((((WorkingSet_sizes[0] + WorkingSet_sizes[1]) + WorkingSet_sizes[3]) +
          WorkingSet_sizes[4]) +
         WorkingSet_sizes[2]) -
        1;
    for (idx = 0; idx <= mAll; idx++) {
      workspace[idx] = lambda[idx];
      lambda[idx] = 0.0F;
    }

    mAll = 0;
    idx = 0;
    while ((idx + 1 <= WorkingSet_nActiveConstr) &&
           (WorkingSet_Wid[idx] <= 2)) {
      if (WorkingSet_Wid[idx] == 1) {
        idxOffset = 1;
      } else {
        idxOffset = WorkingSet_isActiveIdx[1];
      }

      lambda[(idxOffset + WorkingSet_Wlocalidx[idx]) - 2] = workspace[mAll];
      mAll++;
      idx++;
    }

    while (idx + 1 <= WorkingSet_nActiveConstr) {
      switch (WorkingSet_Wid[idx]) {
        case 3:
          idxOffset = WorkingSet_isActiveIdx[2];
          break;

        case 4:
          idxOffset = WorkingSet_isActiveIdx[3];
          break;

        default:
          idxOffset = WorkingSet_isActiveIdx[4];
          break;
      }

      lambda[(idxOffset + WorkingSet_Wlocalidx[idx]) - 2] = workspace[mAll];
      mAll++;
      idx++;
    }
  }
}

/*
 * Arguments    : const float pi4fvek_c[5]
 *                const float mmvek[10]
 *                const bool valid_freq[5]
 *                const float x[4]
 * Return Type  : float
 */
__device__ float sqp_cuda_anonFcn1(const float pi4fvek_c[5],
                                   const float mmvek[10],
                                   const bool valid_freq[5], const float x[4]) {
  float d_x[20];
  float y[20];
  float b_x[10];
  float c_x[10];
  float b_f1;
  float f;
  float varargout_1;
  int b_k;
  int k;

  k = 0;
  f = x[2];
  b_f1 = x[3];
  for (b_k = 0; b_k < 5; b_k++) {
    float b_f2;
    b_f2 = pi4fvek_c[b_k];
    b_x[k] = f * b_f2;
    b_x[k + 1] = b_f1 * b_f2;
    k += 2;
  }

  /*  nF x K*nPts */
  /*  K*nPts x nPts */
  k = 0;
  f = x[0];
  b_f1 = x[1];
  for (b_k = 0; b_k < 10; b_k++) {
    y[k] = f;
    y[k + 1] = b_f1;
    c_x[b_k] = b_x[b_k];
    k += 2;
  }

  k = 0;
  for (b_k = 0; b_k < 5; b_k++) {
    f = __cosf(c_x[k]);
    c_x[k] = f;
    b_f1 = __sinf(b_x[k]);
    b_x[k] = b_f1;
    d_x[k] = f * y[k];
    d_x[k + 10] = b_f1 * y[k + 10];
    f = __cosf(c_x[k + 1]);
    c_x[k + 1] = f;
    b_f1 = __sinf(b_x[k + 1]);
    b_x[k + 1] = b_f1;
    d_x[k + 1] = f * y[k + 1];
    d_x[k + 11] = b_f1 * y[k + 11];
    k += 2;
  }

  k = 0;
  for (b_k = 0; b_k < 10; b_k++) {
    b_x[b_k] = d_x[k] + d_x[k + 1];
    k += 2;
  }

  /*  nF x nPts */
  varargout_1 = 0.0F;
  for (k = 0; k < 5; k++) {
    if (valid_freq[k]) {
      b_k = 2 * k + 1;
      float d1 = b_x[k] - mmvek[k];
      float d2 = b_x[b_k] - mmvek[b_k];
      varargout_1 += d1 * d1 + d2 * d2;
    }
  }

  /*  mmdev2=sum((mmest([valid_freq;valid_freq])-mmvek([valid_freq;valid_freq])*ones(1,nPts,'single')).^2)';
   * %  nPts x 1 */
  return varargout_1;
}

/*
 * Arguments    : int *b_STEP_TYPE
 *                float Hessian[16]
 *                const float lb[4]
 *                const float ub[4]
 *                d_struct_T *TrialState
 *                i_struct_T *MeritFunction
 *                b_struct_T *memspace
 *                e_struct_T *WorkingSet
 *                c_struct_T *QRManager
 *                k_struct_T *CholManager
 *                j_struct_T *QPObjective
 *                g_struct_T qpoptions
 * Return Type  : bool
 */
__device__ bool step(int *b_STEP_TYPE, float Hessian[16], const float lb[4],
                     const float ub[4], d_struct_T *TrialState,
                     i_struct_T *MeritFunction, b_struct_T *memspace,
                     e_struct_T *WorkingSet, c_struct_T *QRManager,
                     k_struct_T *CholManager, j_struct_T *QPObjective,
                     g_struct_T qpoptions) {
  float fv[5];
  float oldDirIdx;
  float penaltyParamTrial;
  int idx;
  int idxIneqOffset;
  int idx_Partition;
  int idx_global;
  int idx_lower;
  int idx_upper;
  int mLB;
  int nVar_tmp_tmp;
  bool checkBoundViolation;
  bool stepSuccess;
  stepSuccess = true;
  checkBoundViolation = true;
  nVar_tmp_tmp = WorkingSet->nVar - 1;
  if (*b_STEP_TYPE != 3) {
    if (nVar_tmp_tmp >= 0) {
      (void)memcpy(&TrialState->xstar[0], &TrialState->xstarsqp[0],
                   (unsigned int)(int)(nVar_tmp_tmp + 1) * sizeof(float));
    }
  } else if (nVar_tmp_tmp >= 0) {
    (void)memcpy(&TrialState->searchDir[0], &TrialState->xstar[0],
                 (unsigned int)(int)(nVar_tmp_tmp + 1) * sizeof(float));
  } else {
    /* no actions */
  }

  int exitg1;
  bool guard1;
  do {
    exitg1 = 0;
    guard1 = false;
    switch (*b_STEP_TYPE) {
      case 1:
        for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
          fv[idxIneqOffset] = TrialState->grad[idxIneqOffset];
        }

        driver(Hessian, fv, TrialState, memspace, WorkingSet, QRManager,
               CholManager, QPObjective, qpoptions, qpoptions);
        if (TrialState->state > 0) {
          penaltyParamTrial = MeritFunction->penaltyParam;
          oldDirIdx = MeritFunction->linearizedConstrViol;
          MeritFunction->linearizedConstrViol = 0.0F;
          if ((oldDirIdx > 1.1920929E-7F) && (TrialState->fstar > 0.0F)) {
            if (TrialState->sqpFval == 0.0F) {
              penaltyParamTrial = 1.0F;
            } else {
              penaltyParamTrial = 1.5F;
            }

            penaltyParamTrial =
                penaltyParamTrial * TrialState->fstar / oldDirIdx;
          }

          if (penaltyParamTrial < MeritFunction->penaltyParam) {
            MeritFunction->phi = TrialState->sqpFval;
            if ((MeritFunction->initFval +
                 penaltyParamTrial * (MeritFunction->initConstrViolationEq +
                                      MeritFunction->initConstrViolationIneq)) -
                    TrialState->sqpFval >
                (float)MeritFunction->nPenaltyDecreases *
                    MeritFunction->threshold) {
              MeritFunction->nPenaltyDecreases++;
              if (MeritFunction->nPenaltyDecreases * 2 >
                  TrialState->sqpIterations) {
                MeritFunction->threshold *= 10.0F;
              }

              MeritFunction->penaltyParam =
                  fmaxf(penaltyParamTrial, 1.1920929E-7F);
            } else {
              MeritFunction->phi = TrialState->sqpFval;
            }
          } else {
            MeritFunction->penaltyParam =
                fmaxf(penaltyParamTrial, 1.1920929E-7F);
            MeritFunction->phi = TrialState->sqpFval;
          }

          MeritFunction->phiPrimePlus = fminf(TrialState->fstar, 0.0F);
        }

        sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                     WorkingSet->sizes, WorkingSet->isActiveIdx,
                     WorkingSet->Wid, WorkingSet->Wlocalidx,
                     memspace->workspace_float);
        if (WorkingSet->mEqRemoved > 0) {
          idxIneqOffset = (WorkingSet->sizes[0] + TrialState->iNonEq0) - 1;
          idx_global = TrialState->mNonlinEq;
          for (idx = 0; idx < idx_global; idx++) {
            WorkingSet->Wlocalidx[idxIneqOffset + idx] =
                TrialState->iNonEq0 + idx;
          }
        }

        if ((TrialState->state <= 0) && (TrialState->state != -6)) {
          *b_STEP_TYPE = 2;
        } else {
          if (nVar_tmp_tmp >= 0) {
            (void)memcpy(&TrialState->delta_x[0], &TrialState->xstar[0],
                         (unsigned int)(int)(nVar_tmp_tmp + 1) * sizeof(float));
          }

          guard1 = true;
        }
        break;

      case 2:
        idxIneqOffset = WorkingSet->nWConstr[0] + WorkingSet->nWConstr[1];
        idx_lower = idxIneqOffset + 1;
        idx_upper = WorkingSet->nActiveConstr;
        for (idx_global = idx_lower; idx_global <= idx_upper; idx_global++) {
          WorkingSet->isActiveConstr
              [(WorkingSet->isActiveIdx[WorkingSet->Wid[idx_global - 1] - 1] +
                WorkingSet->Wlocalidx[idx_global - 1]) -
               2] = false;
        }

        WorkingSet->nWConstr[2] = 0;
        WorkingSet->nWConstr[3] = 0;
        WorkingSet->nWConstr[4] = 0;
        WorkingSet->nActiveConstr = idxIneqOffset;
        for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
          fv[idxIneqOffset] = TrialState->xstar[idxIneqOffset];
        }

        mLB = WorkingSet->sizes[3];
        idx_lower = WorkingSet->sizes[4];
        for (idx = 0; idx < mLB; idx++) {
          idxIneqOffset = WorkingSet->indexLB[idx] - 1;
          if (-fv[WorkingSet->indexLB[idx] - 1] >
              WorkingSet->lb[WorkingSet->indexLB[idx] - 1]) {
            fv[WorkingSet->indexLB[idx] - 1] = (WorkingSet->ub[idxIneqOffset] -
                                                WorkingSet->lb[idxIneqOffset]) /
                                               2.0F;
          }
        }

        for (idx = 0; idx < idx_lower; idx++) {
          idxIneqOffset = WorkingSet->indexUB[idx] - 1;
          if (fv[WorkingSet->indexUB[idx] - 1] >
              WorkingSet->ub[WorkingSet->indexUB[idx] - 1]) {
            fv[WorkingSet->indexUB[idx] - 1] = (WorkingSet->ub[idxIneqOffset] -
                                                WorkingSet->lb[idxIneqOffset]) /
                                               2.0F;
          }
        }

        for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
          TrialState->xstar[idxIneqOffset] = fv[idxIneqOffset];
        }

        relaxed(Hessian, TrialState->grad, TrialState, MeritFunction, memspace,
                WorkingSet, QRManager, CholManager, QPObjective, &qpoptions);
        if (nVar_tmp_tmp >= 0) {
          (void)memcpy(&TrialState->delta_x[0], &TrialState->xstar[0],
                       (unsigned int)(int)(nVar_tmp_tmp + 1) * sizeof(float));
        }

        guard1 = true;
        break;

      default: {
        int mConstrMax;
        int nWIneq_old;
        int nWLower_old;
        int nWUpper_old;
        nWIneq_old = WorkingSet->nWConstr[2];
        nWLower_old = WorkingSet->nWConstr[3];
        nWUpper_old = WorkingSet->nWConstr[4];
        idx_global = WorkingSet->nVar - 1;
        mConstrMax = WorkingSet->mConstrMax - 1;
        for (idx_lower = 0; idx_lower <= idx_global; idx_lower++) {
          TrialState->xstarsqp[idx_lower] = TrialState->xstarsqp_old[idx_lower];
          TrialState->socDirection[idx_lower] = TrialState->xstar[idx_lower];
        }

        if (mConstrMax >= 0) {
          (void)memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambda[0],
                       (unsigned int)(int)(mConstrMax + 1) * sizeof(float));
        }

        idxIneqOffset = WorkingSet->isActiveIdx[2];
        if (WorkingSet->sizes[2] > 0) {
          idx_lower = WorkingSet->sizes[2] + 1;
          idx_upper = (WorkingSet->sizes[2] + WorkingSet->sizes[3]) + 1;
          mLB = WorkingSet->nActiveConstr;
          for (idx = idxIneqOffset; idx <= mLB; idx++) {
            switch (WorkingSet->Wid[idx - 1]) {
              case 3:
                /* A check that is always false is detected at compile-time.
             * Eliminating code that follows. */
                break;

              case 4:
                idx_Partition = idx_lower;
                idx_lower++;
                break;

              default:
                idx_Partition = idx_upper;
                idx_upper++;
                break;
            }

            TrialState->workingset_old[idx_Partition - 1] =
                WorkingSet->Wlocalidx[idx - 1];
          }
        }

        if (idx_global >= 0) {
          (void)memcpy(&TrialState->xstar[0], &TrialState->xstarsqp[0],
                       (unsigned int)(int)(idx_global + 1) * sizeof(float));
        }

        for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
          fv[idxIneqOffset] = TrialState->grad[idxIneqOffset];
        }

        driver(Hessian, fv, TrialState, memspace, WorkingSet, QRManager,
               CholManager, QPObjective, qpoptions, qpoptions);
        for (idx = 0; idx <= idx_global; idx++) {
          penaltyParamTrial = TrialState->socDirection[idx];
          oldDirIdx = penaltyParamTrial;
          penaltyParamTrial = TrialState->xstar[idx] - penaltyParamTrial;
          TrialState->socDirection[idx] = penaltyParamTrial;
          TrialState->xstar[idx] = oldDirIdx;
        }

        stepSuccess = (b_xnrm2(idx_global + 1, TrialState->socDirection) <=
                       2.0F * b_xnrm2(idx_global + 1, TrialState->xstar));
        idxIneqOffset = WorkingSet->sizes[2];
        mLB = WorkingSet->sizes[3];
        if ((WorkingSet->sizes[2] > 0) && (!stepSuccess)) {
          idx_lower = (WorkingSet->nWConstr[0] + WorkingSet->nWConstr[1]) + 1;
          idx_upper = WorkingSet->nActiveConstr;
          for (idx_global = idx_lower; idx_global <= idx_upper; idx_global++) {
            WorkingSet->isActiveConstr
                [(WorkingSet->isActiveIdx[WorkingSet->Wid[idx_global - 1] - 1] +
                  WorkingSet->Wlocalidx[idx_global - 1]) -
                 2] = false;
          }

          WorkingSet->nWConstr[2] = 0;
          WorkingSet->nWConstr[3] = 0;
          WorkingSet->nWConstr[4] = 0;
          WorkingSet->nActiveConstr =
              WorkingSet->nWConstr[0] + WorkingSet->nWConstr[1];
          for (idx = 0; idx < nWIneq_old; idx++) {
            addAineqConstr(WorkingSet);
          }

          for (idx = 0; idx < nWLower_old; idx++) {
            addBoundToActiveSetMatrix_(
                WorkingSet, 4, TrialState->workingset_old[idx + idxIneqOffset]);
          }

          for (idx = 0; idx < nWUpper_old; idx++) {
            addBoundToActiveSetMatrix_(
                WorkingSet, 5,
                TrialState->workingset_old[(idx + idxIneqOffset) + mLB]);
          }
        }

        if (!stepSuccess) {
          if (mConstrMax >= 0) {
            (void)memcpy(&TrialState->lambda[0], &TrialState->lambdaStopTest[0],
                         (unsigned int)(int)(mConstrMax + 1) * sizeof(float));
          }
        } else {
          sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                       WorkingSet->sizes, WorkingSet->isActiveIdx,
                       WorkingSet->Wid, WorkingSet->Wlocalidx,
                       memspace->workspace_float);
        }

        checkBoundViolation = stepSuccess;
        if (stepSuccess && (TrialState->state != -6)) {
          for (idx = 0; idx <= nVar_tmp_tmp; idx++) {
            TrialState->delta_x[idx] =
                TrialState->xstar[idx] + TrialState->socDirection[idx];
          }
        }

        guard1 = true;
      } break;
    }

    if (guard1) {
      if (TrialState->state != -6) {
        exitg1 = 1;
      } else {
        penaltyParamTrial = fmaxf(
            1.1920929E-7F,
            fmaxf(fmaxf(fmaxf(fmaxf(0.0F, fabsf(TrialState->grad[0])),
                              fabsf(TrialState->grad[1])),
                        fabsf(TrialState->grad[2])),
                  fabsf(TrialState->grad[3])) /
                fmaxf(fmaxf(fmaxf(fmaxf(1.0F, fabsf(TrialState->xstar[0])),
                                  fabsf(TrialState->xstar[1])),
                            fabsf(TrialState->xstar[2])),
                      fabsf(TrialState->xstar[3])));
        for (idx_upper = 0; idx_upper < 4; idx_upper++) {
          idxIneqOffset = 4 * idx_upper;
          for (idx_lower = 0; idx_lower < idx_upper; idx_lower++) {
            Hessian[idxIneqOffset + idx_lower] = 0.0F;
          }

          idxIneqOffset = idx_upper + 4 * idx_upper;
          Hessian[idxIneqOffset] = penaltyParamTrial;
          idx_lower = 2 - idx_upper;
          if (idx_lower >= 0) {
            (void)memset(&Hessian[idxIneqOffset + 1], 0,
                         (unsigned int)(int)(((idx_lower + idxIneqOffset) -
                                              idxIneqOffset) +
                                             1) *
                             sizeof(float));
          }
        }
      }
    }
  } while (exitg1 == 0);

  if (checkBoundViolation) {
    mLB = WorkingSet->sizes[3];
    idx_lower = WorkingSet->sizes[4];
    for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
      fv[idxIneqOffset] = TrialState->delta_x[idxIneqOffset];
    }

    for (idx = 0; idx < mLB; idx++) {
      oldDirIdx = fv[WorkingSet->indexLB[idx] - 1];
      penaltyParamTrial =
          (TrialState->xstarsqp[WorkingSet->indexLB[idx] - 1] + oldDirIdx) -
          lb[WorkingSet->indexLB[idx] - 1];
      if (penaltyParamTrial < 0.0F) {
        fv[WorkingSet->indexLB[idx] - 1] = oldDirIdx - penaltyParamTrial;
        TrialState->xstar[WorkingSet->indexLB[idx] - 1] -= penaltyParamTrial;
      }
    }

    for (idx = 0; idx < idx_lower; idx++) {
      oldDirIdx = fv[WorkingSet->indexUB[idx] - 1];
      penaltyParamTrial = (ub[WorkingSet->indexUB[idx] - 1] -
                           TrialState->xstarsqp[WorkingSet->indexUB[idx] - 1]) -
                          oldDirIdx;
      if (penaltyParamTrial < 0.0F) {
        fv[WorkingSet->indexUB[idx] - 1] = oldDirIdx + penaltyParamTrial;
        TrialState->xstar[WorkingSet->indexUB[idx] - 1] += penaltyParamTrial;
      }
    }

    for (idxIneqOffset = 0; idxIneqOffset < 5; idxIneqOffset++) {
      TrialState->delta_x[idxIneqOffset] = fv[idxIneqOffset];
    }
  }

  return stepSuccess;
}

/*
 * Arguments    : b_struct_T *memspace
 *                i_struct_T *MeritFunction
 *                e_struct_T *WorkingSet
 *                d_struct_T *TrialState
 *                c_struct_T *QRManager
 *                const float lb[4]
 *                const float ub[4]
 *                bool *Flags_fevalOK
 *                bool *Flags_done
 *                bool *Flags_stepAccepted
 *                bool *Flags_failedLineSearch
 *                int *Flags_stepType
 * Return Type  : bool
 */
__device__ bool test_exit(b_struct_T *memspace, i_struct_T *MeritFunction,
                          e_struct_T *WorkingSet, d_struct_T *TrialState,
                          c_struct_T *QRManager, const float lb[4],
                          const float ub[4], bool *Flags_fevalOK,
                          bool *Flags_done, bool *Flags_stepAccepted,
                          bool *Flags_failedLineSearch, int *Flags_stepType,
                          float optimality_tolerance) {
  float f;
  float optimRelativeFactor;
  float s;
  float smax;
  int b_idx_max;
  int k;
  int mIneq;
  int mLB;
  int mLambda;
  int mLambda_tmp;
  int mUB;
  int nVar;
  bool Flags_gradOK;
  bool exitg1;
  bool isFeasible;
  *Flags_fevalOK = true;
  *Flags_done = false;
  *Flags_stepAccepted = false;
  *Flags_failedLineSearch = false;
  *Flags_stepType = 1;
  nVar = WorkingSet->nVar;
  mIneq = WorkingSet->sizes[2];
  mLB = WorkingSet->sizes[3];
  mUB = WorkingSet->sizes[4];
  mLambda_tmp = WorkingSet->sizes[0] + WorkingSet->sizes[1];
  mLambda = (((mLambda_tmp + WorkingSet->sizes[2]) + WorkingSet->sizes[3]) +
             WorkingSet->sizes[4]) -
            1;
  if (mLambda >= 0) {
    (void)memcpy(&TrialState->lambdaStopTest[0], &TrialState->lambdasqp[0],
                 (unsigned int)(int)(mLambda + 1) * sizeof(float));
  }

  computeGradLag(TrialState->gradLag, WorkingSet->nVar, TrialState->grad,
                 WorkingSet->sizes[2], WorkingSet->sizes[1],
                 WorkingSet->indexFixed, WorkingSet->sizes[0],
                 WorkingSet->indexLB, WorkingSet->sizes[3], WorkingSet->indexUB,
                 WorkingSet->sizes[4], TrialState->lambdaStopTest);
  if (WorkingSet->nVar < 1) {
    b_idx_max = 0;
  } else {
    b_idx_max = 1;
    if (WorkingSet->nVar > 1) {
      smax = fabsf(TrialState->grad[0]);
      for (k = 2; k <= nVar; k++) {
        s = fabsf(TrialState->grad[k - 1]);
        if (s > smax) {
          b_idx_max = k;
          smax = s;
        }
      }
    }
  }

  optimRelativeFactor = fmaxf(1.0F, fabsf(TrialState->grad[b_idx_max - 1]));
  if (optimRelativeFactor >= 3.402823466E+38F) {
    optimRelativeFactor = 1.0F;
  }

  MeritFunction->nlpPrimalFeasError = computePrimalFeasError(
      TrialState->xstarsqp, WorkingSet->indexLB, WorkingSet->sizes[3], lb,
      WorkingSet->indexUB, WorkingSet->sizes[4], ub);
  if (TrialState->sqpIterations == 0) {
    MeritFunction->feasRelativeFactor =
        fmaxf(1.0F, MeritFunction->nlpPrimalFeasError);
  }

  isFeasible = (MeritFunction->nlpPrimalFeasError <=
                0.001F * MeritFunction->feasRelativeFactor);
  Flags_gradOK = true;
  s = 0.0F;
  b_idx_max = 0;
  exitg1 = false;
  while ((!exitg1) && (b_idx_max <= nVar - 1)) {
    f = fabsf(TrialState->gradLag[b_idx_max]);
    Flags_gradOK = (f < 3.402823466E+38F);
    if (!Flags_gradOK) {
      exitg1 = true;
    } else {
      s = fmaxf(s, f);
      b_idx_max++;
    }
  }

  MeritFunction->nlpDualFeasError = s;
  if (!Flags_gradOK) {
    *Flags_done = true;
    if (isFeasible) {
      TrialState->sqpExitFlag = 2;
    } else {
      TrialState->sqpExitFlag = -2;
    }
  } else {
    MeritFunction->nlpComplError = computeComplError(
        TrialState->xstarsqp, WorkingSet->sizes[2], WorkingSet->indexLB,
        WorkingSet->sizes[3], lb, WorkingSet->indexUB, WorkingSet->sizes[4], ub,
        TrialState->lambdaStopTest, mLambda_tmp + 1);
    MeritFunction->b_firstOrderOpt = fmaxf(s, MeritFunction->nlpComplError);
    if (TrialState->sqpIterations > 1) {
      float nlpDualFeasErrorTmp;
      b_computeGradLag(memspace->workspace_float, WorkingSet->nVar,
                       TrialState->grad, WorkingSet->sizes[2],
                       WorkingSet->sizes[1], WorkingSet->indexFixed,
                       WorkingSet->sizes[0], WorkingSet->indexLB,
                       WorkingSet->sizes[3], WorkingSet->indexUB,
                       WorkingSet->sizes[4], TrialState->lambdaStopTestPrev);
      nlpDualFeasErrorTmp = 0.0F;
      b_idx_max = 0;
      exitg1 = false;
      while ((!exitg1) && (b_idx_max <= nVar - 1)) {
        f = fabsf(memspace->workspace_float[b_idx_max]);
        if (f >= 3.402823466E+38F) {
          exitg1 = true;
        } else {
          nlpDualFeasErrorTmp = fmaxf(nlpDualFeasErrorTmp, f);
          b_idx_max++;
        }
      }

      smax = computeComplError(TrialState->xstarsqp, WorkingSet->sizes[2],
                               WorkingSet->indexLB, WorkingSet->sizes[3], lb,
                               WorkingSet->indexUB, WorkingSet->sizes[4], ub,
                               TrialState->lambdaStopTestPrev, mLambda_tmp + 1);
      if ((nlpDualFeasErrorTmp < s) && (smax < MeritFunction->nlpComplError)) {
        MeritFunction->nlpDualFeasError = nlpDualFeasErrorTmp;
        MeritFunction->nlpComplError = smax;
        MeritFunction->b_firstOrderOpt = fmaxf(nlpDualFeasErrorTmp, smax);
        if (mLambda >= 0) {
          (void)memcpy(&TrialState->lambdaStopTest[0],
                       &TrialState->lambdaStopTestPrev[0],
                       (unsigned int)(int)(mLambda + 1) * sizeof(float));
        }
      } else if (mLambda >= 0) {
        (void)memcpy(&TrialState->lambdaStopTestPrev[0],
                     &TrialState->lambdaStopTest[0],
                     (unsigned int)(int)(mLambda + 1) * sizeof(float));
      } else {
        /* no actions */
      }
    } else if (mLambda >= 0) {
      (void)memcpy(&TrialState->lambdaStopTestPrev[0],
                   &TrialState->lambdaStopTest[0],
                   (unsigned int)(int)(mLambda + 1) * sizeof(float));
    } else {
      /* no actions */
    }

    if (isFeasible &&
        (MeritFunction->nlpDualFeasError <=
         optimality_tolerance * optimRelativeFactor) &&
        (MeritFunction->nlpComplError <=
         optimality_tolerance * optimRelativeFactor)) {
      *Flags_done = true;
      TrialState->sqpExitFlag = 1;
    } else if (isFeasible && ((double)TrialState->sqpFval < -1.0E+20)) {
      *Flags_done = true;
      TrialState->sqpExitFlag = -3;
    } else {
      bool guard1;
      guard1 = false;
      if (TrialState->sqpIterations > 0) {
        bool dxTooSmall;
        dxTooSmall = true;
        b_idx_max = 0;
        exitg1 = false;
        while ((!exitg1) && (b_idx_max <= nVar - 1)) {
          if (0.0001F * fmaxf(1.0F, fabsf(TrialState->xstarsqp[b_idx_max])) <=
              fabsf(TrialState->delta_x[b_idx_max])) {
            dxTooSmall = false;
            exitg1 = true;
          } else {
            b_idx_max++;
          }
        }

        if (dxTooSmall) {
          if (!isFeasible) {
            *Flags_stepType = 2;
            guard1 = true;
          } else if (WorkingSet->nActiveConstr == 0) {
            *Flags_done = true;
            TrialState->sqpExitFlag = 2;
          } else {
            if (TrialState->mNonlinEq + TrialState->mNonlinIneq > 0) {
              updateWorkingSetForNewQP(
                  TrialState->xstarsqp, WorkingSet, WorkingSet->sizes[1],
                  WorkingSet->sizes[3], lb, WorkingSet->sizes[4], ub,
                  WorkingSet->sizes[0]);
            }

            computeLambdaLSQ(nVar, WorkingSet->nActiveConstr, QRManager,
                             WorkingSet->ATwset, WorkingSet->ldA,
                             TrialState->grad, TrialState->lambda,
                             memspace->workspace_float);
            sortLambdaQP(TrialState->lambda, WorkingSet->nActiveConstr,
                         WorkingSet->sizes, WorkingSet->isActiveIdx,
                         WorkingSet->Wid, WorkingSet->Wlocalidx,
                         memspace->workspace_float);
            b_computeGradLag(memspace->workspace_float, nVar, TrialState->grad,
                             mIneq, WorkingSet->sizes[1],
                             WorkingSet->indexFixed, WorkingSet->sizes[0],
                             WorkingSet->indexLB, mLB, WorkingSet->indexUB, mUB,
                             TrialState->lambda);
            smax = 0.0F;
            b_idx_max = 0;
            exitg1 = false;
            while ((!exitg1) && (b_idx_max <= nVar - 1)) {
              f = fabsf(memspace->workspace_float[b_idx_max]);
              if (f >= 3.402823466E+38F) {
                exitg1 = true;
              } else {
                smax = fmaxf(smax, f);
                b_idx_max++;
              }
            }

            s = computeComplError(TrialState->xstarsqp, mIneq,
                                  WorkingSet->indexLB, mLB, lb,
                                  WorkingSet->indexUB, mUB, ub,
                                  TrialState->lambda, mLambda_tmp + 1);
            f = fmaxf(smax, s);
            if (f <= fmaxf(MeritFunction->nlpDualFeasError,
                           MeritFunction->nlpComplError)) {
              MeritFunction->nlpDualFeasError = smax;
              MeritFunction->nlpComplError = s;
              MeritFunction->b_firstOrderOpt = f;
              if (mLambda >= 0) {
                (void)memcpy(&TrialState->lambdaStopTest[0],
                             &TrialState->lambda[0],
                             (unsigned int)(int)(mLambda + 1) * sizeof(float));
              }
            }

            if ((MeritFunction->nlpDualFeasError <=
                 optimality_tolerance * optimRelativeFactor) &&
                (MeritFunction->nlpComplError <=
                 optimality_tolerance * optimRelativeFactor)) {
              TrialState->sqpExitFlag = 1;
            } else {
              TrialState->sqpExitFlag = 2;
            }

            *Flags_done = true;
            guard1 = true;
          }
        } else {
          guard1 = true;
        }
      } else {
        guard1 = true;
      }

      if (guard1) {
        if (TrialState->sqpIterations >= kMaxIter) {
          *Flags_done = true;
          TrialState->sqpExitFlag = 0;
        } else if (TrialState->FunctionEvaluations >= 4 * kMaxIter) {
          *Flags_done = true;
          TrialState->sqpExitFlag = 0;
        } else {
          /* no actions */
        }
      }
    }
  }

  return Flags_gradOK;
}

/*
 * Arguments    : const float xk[4]
 *                e_struct_T *WorkingSet
 *                int mEq
 *                int mLB
 *                const float lb[4]
 *                int mUB
 *                const float ub[4]
 *                int mFixed
 * Return Type  : void
 */
__device__ void updateWorkingSetForNewQP(const float xk[4],
                                         e_struct_T *WorkingSet, int mEq,
                                         int mLB, const float lb[4], int mUB,
                                         const float ub[4], int mFixed) {
  int b_i;
  int idx;
  for (idx = 0; idx < mLB; idx++) {
    WorkingSet->lb[WorkingSet->indexLB[idx] - 1] =
        -lb[WorkingSet->indexLB[idx] - 1] + xk[WorkingSet->indexLB[idx] - 1];
  }

  for (idx = 0; idx < mUB; idx++) {
    WorkingSet->ub[WorkingSet->indexUB[idx] - 1] =
        ub[WorkingSet->indexUB[idx] - 1] - xk[WorkingSet->indexUB[idx] - 1];
  }

  for (idx = 0; idx < mFixed; idx++) {
    float f;
    f = ub[WorkingSet->indexFixed[idx] - 1] -
        xk[WorkingSet->indexFixed[idx] - 1];
    WorkingSet->ub[WorkingSet->indexFixed[idx] - 1] = f;
    WorkingSet->bwset[idx] = f;
  }

  b_i = mFixed + mEq;
  if (WorkingSet->nActiveConstr > b_i) {
    int ineqStart;
    ineqStart = b_i + 1;
    if (ineqStart < 1) {
      ineqStart = 1;
    }

    b_i = WorkingSet->nActiveConstr;
    for (idx = ineqStart; idx <= b_i; idx++) {
      switch (WorkingSet->Wid[idx - 1]) {
        case 4:
          WorkingSet->bwset[idx - 1] =
              WorkingSet
                  ->lb[WorkingSet->indexLB[WorkingSet->Wlocalidx[idx - 1] - 1] -
                       1];
          break;

        case 5:
          WorkingSet->bwset[idx - 1] =
              WorkingSet
                  ->ub[WorkingSet->indexUB[WorkingSet->Wlocalidx[idx - 1] - 1] -
                       1];
          break;

        default:
          /* A check that is always false is detected at compile-time. Eliminating
         * code that follows. */
          break;
      }
    }
  }
}

/*
 * Arguments    : int m
 *                int n
 *                int k
 *                const float b_A[16]
 *                int b_lda
 *                const float b_B[81]
 *                int ib0
 *                int ldb
 *                float C[45]
 * Return Type  : void
 */
__device__ void xgemm(int m, int n, int k, const float b_A[16], int b_lda,
                      const float b_B[81], int ib0, int ldb, float C[45]) {
  int b_ib;
  int cr;
  int ic;
  if ((m != 0) && (n != 0)) {
    int b_i;
    int br;
    int i1;
    int lastColC;
    br = ib0;
    lastColC = 9 * (n - 1);
    for (cr = 0; cr <= lastColC; cr += 9) {
      b_i = cr + 1;
      i1 = cr + m;
      if (b_i <= i1) {
        (void)memset(&C[b_i + -1], 0,
                     (unsigned int)(int)((i1 - b_i) + 1) * sizeof(float));
      }
    }

    for (cr = 0; cr <= lastColC; cr += 9) {
      int ar;
      ar = -1;
      b_i = br + k;
      for (b_ib = br; b_ib < b_i; b_ib++) {
        int i2;
        i1 = cr + 1;
        i2 = cr + m;
        for (ic = i1; ic <= i2; ic++) {
          C[ic - 1] += b_B[b_ib - 1] * b_A[(ar + ic) - cr];
        }

        ar += b_lda;
      }

      br += ldb;
    }
  }
}

/*
 * Arguments    : float b_A[81]
 *                int m
 *                int n
 *                int jpvt[9]
 *                float tau[9]
 * Return Type  : void
 */
__device__ void xgeqp3(float b_A[81], int m, int n, int jpvt[9], float tau[9]) {
  float temp;
  int b_i;
  int k;
  int minmn_tmp;
  int pvt;
  if (m <= n) {
    minmn_tmp = m;
  } else {
    minmn_tmp = n;
  }

  for (b_i = 0; b_i < 9; b_i++) {
    tau[b_i] = 0.0F;
  }

  if (minmn_tmp < 1) {
    for (pvt = 0; pvt < n; pvt++) {
      jpvt[pvt] = pvt + 1;
    }
  } else {
    int c_i;
    int ix;
    int iy;
    int nfxd;
    int temp_tmp;
    nfxd = 0;
    for (pvt = 0; pvt < n; pvt++) {
      if (jpvt[pvt] != 0) {
        nfxd++;
        if (pvt + 1 != nfxd) {
          ix = pvt * 9;
          iy = (nfxd - 1) * 9;
          for (k = 0; k < m; k++) {
            temp_tmp = ix + k;
            temp = b_A[temp_tmp];
            c_i = iy + k;
            b_A[temp_tmp] = b_A[c_i];
            b_A[c_i] = temp;
          }

          jpvt[pvt] = jpvt[nfxd - 1];
          jpvt[nfxd - 1] = pvt + 1;
        } else {
          jpvt[pvt] = pvt + 1;
        }
      } else {
        jpvt[pvt] = pvt + 1;
      }
    }

    if (nfxd > minmn_tmp) {
      nfxd = minmn_tmp;
    }

    qrf(b_A, m, n, nfxd, tau);
    if (nfxd < minmn_tmp) {
      float vn1[9];
      float vn2[9];
      float work[9];
      float f;
      for (b_i = 0; b_i < 9; b_i++) {
        work[b_i] = 0.0F;
        vn1[b_i] = 0.0F;
        vn2[b_i] = 0.0F;
      }

      c_i = nfxd + 1;
      for (pvt = c_i; pvt <= n; pvt++) {
        f = xnrm2(m - nfxd, b_A, (nfxd + (pvt - 1) * 9) + 1);
        vn1[pvt - 1] = f;
        vn2[pvt - 1] = f;
      }

      for (b_i = c_i; b_i <= minmn_tmp; b_i++) {
        float s;
        int b_ii;
        int ip1;
        int mmi;
        int nmi;
        ip1 = b_i + 1;
        nfxd = (b_i - 1) * 9;
        b_ii = (nfxd + b_i) - 1;
        nmi = (n - b_i) + 1;
        mmi = m - b_i;
        if (nmi < 1) {
          iy = -2;
        } else {
          iy = -1;
          if (nmi > 1) {
            temp = fabsf(vn1[b_i - 1]);
            for (k = 2; k <= nmi; k++) {
              s = fabsf(vn1[(b_i + k) - 2]);
              if (s > temp) {
                iy = k - 2;
                temp = s;
              }
            }
          }
        }

        pvt = b_i + iy;
        if (pvt + 1 != b_i) {
          ix = pvt * 9;
          for (k = 0; k < m; k++) {
            temp_tmp = ix + k;
            temp = b_A[temp_tmp];
            iy = nfxd + k;
            b_A[temp_tmp] = b_A[iy];
            b_A[iy] = temp;
          }

          iy = jpvt[pvt];
          jpvt[pvt] = jpvt[b_i - 1];
          jpvt[b_i - 1] = iy;
          vn1[pvt] = vn1[b_i - 1];
          vn2[pvt] = vn2[b_i - 1];
        }

        if (b_i < m) {
          temp = b_A[b_ii];
          f = xzlarfg(mmi + 1, &temp, b_A, b_ii + 2);
          tau[b_i - 1] = f;
          b_A[b_ii] = temp;
        } else {
          f = 0.0F;
          tau[b_i - 1] = 0.0F;
        }

        if (b_i < n) {
          temp = b_A[b_ii];
          b_A[b_ii] = 1.0F;
          xzlarf(mmi + 1, nmi - 1, b_ii + 1, f, b_A, b_ii + 10, work);
          b_A[b_ii] = temp;
        }

        for (pvt = ip1; pvt <= n; pvt++) {
          iy = b_i + (pvt - 1) * 9;
          f = vn1[pvt - 1];
          if (f != 0.0F) {
            temp = fabsf(b_A[iy - 1]) / f;
            temp = 1.0F - temp * temp;
            if (temp < 0.0F) {
              temp = 0.0F;
            }

            s = f / vn2[pvt - 1];
            s = temp * (s * s);
            if (s <= 0.000345266977F) {
              if (b_i < m) {
                f = xnrm2(mmi, b_A, iy + 1);
                vn1[pvt - 1] = f;
                vn2[pvt - 1] = f;
              } else {
                vn1[pvt - 1] = 0.0F;
                vn2[pvt - 1] = 0.0F;
              }
            } else {
              vn1[pvt - 1] = f * sqrtf(temp);
            }
          }
        }
      }
    }
  }
}

/*
 * Arguments    : int n
 *                const float x[81]
 *                int ix0
 * Return Type  : float
 */
__device__ float xnrm2(int n, const float x[81], int ix0) {
  float y;
  int k;
  y = 0.0F;
  if (n >= 1) {
    if (n == 1) {
      y = fabsf(x[ix0 - 1]);
    } else {
      float scale;
      int kend;
      scale = 1.29246971E-26F;
      kend = (ix0 + n) - 1;
      for (k = ix0; k <= kend; k++) {
        float absxk;
        absxk = fabsf(x[k - 1]);
        if (absxk > scale) {
          float t;
          t = scale / absxk;
          y = y * t * t + 1.0F;
          scale = absxk;
        } else {
          float t;
          t = absxk / scale;
          y += t * t;
        }
      }

      y = scale * sqrtf(y);
    }
  }

  return y;
}

/*
 * Arguments    : int n
 *                float b_A[81]
 *                int b_lda
 * Return Type  : int
 */
__device__ int xpotrf(int n, float b_A[81], int b_lda) {
  int ia;
  int info;
  int j;
  int k;
  bool exitg1;
  info = 0;
  j = 0;
  exitg1 = false;
  while ((!exitg1) && (j <= n - 1)) {
    float b_c;
    float ssq;
    int idxA1j;
    int idxAjj;
    idxA1j = j * b_lda;
    idxAjj = idxA1j + j;
    ssq = 0.0F;
    if (j >= 1) {
      for (k = 0; k < j; k++) {
        b_c = b_A[idxA1j + k];
        ssq += b_c * b_c;
      }
    }

    ssq = b_A[idxAjj] - ssq;
    if (ssq > 0.0F) {
      ssq = sqrtf(ssq);
      b_A[idxAjj] = ssq;
      if (j + 1 < n) {
        int b_i;
        int ia0;
        int idxAjjp1;
        int nmj;
        nmj = (n - j) - 2;
        ia0 = (idxA1j + b_lda) + 1;
        idxAjjp1 = idxAjj + b_lda;
        if ((j != 0) && (nmj + 1 != 0)) {
          int iac;
          idxAjj = idxAjjp1;
          b_i = ia0 + b_lda * nmj;
          iac = ia0;
          while (((b_lda > 0) && (iac <= b_i)) ||
                 ((b_lda < 0) && (iac >= b_i))) {
            b_c = 0.0F;
            k = (iac + j) - 1;
            for (ia = iac; ia <= k; ia++) {
              b_c += b_A[ia - 1] * b_A[(idxA1j + ia) - iac];
            }

            b_A[idxAjj] -= b_c;
            idxAjj += b_lda;
            iac += b_lda;
          }
        }

        ssq = 1.0F / ssq;
        if (b_lda >= 1) {
          b_i = (idxAjjp1 + b_lda * nmj) + 1;
          k = idxAjjp1 + 1;
          while (((b_lda > 0) && (k <= b_i)) || ((b_lda < 0) && (k >= b_i))) {
            b_A[k - 1] *= ssq;
            k += b_lda;
          }
        }
      }

      j++;
    } else {
      b_A[idxAjj] = ssq;
      info = j + 1;
      exitg1 = true;
    }
  }

  return info;
}

/*
 * Arguments    : float *a
 *                float *b
 *                float *s
 * Return Type  : float
 */
__device__ float xrotg(float *a, float *b, float *s) {
  float absa;
  float absb;
  float b_c;
  float roe;
  float scale;
  roe = *b;
  absa = fabsf(*a);
  absb = fabsf(*b);
  if (absa > absb) {
    roe = *a;
  }

  scale = absa + absb;
  if (scale == 0.0F) {
    *s = 0.0F;
    b_c = 1.0F;
    *a = 0.0F;
    *b = 0.0F;
  } else {
    float ads;
    float bds;
    ads = absa / scale;
    bds = absb / scale;
    scale *= sqrtf(ads * ads + bds * bds);
    if (roe < 0.0F) {
      scale = -scale;
    }

    b_c = *a / scale;
    *s = *b / scale;
    if (absa > absb) {
      *b = *s;
    } else if (b_c != 0.0F) {
      *b = 1.0F / b_c;
    } else {
      *b = 1.0F;
    }

    *a = scale;
  }

  return b_c;
}

/*
 * Arguments    : int m
 *                int n
 *                int iv0
 *                float tau
 *                float C[81]
 *                int ic0
 *                float work[9]
 * Return Type  : void
 */
__device__ void xzlarf(int m, int n, int iv0, float tau, float C[81], int ic0,
                       float work[9]) {
  int b_i;
  int ia;
  int iac;
  int lastc;
  int lastv;
  if (tau != 0.0F) {
    bool exitg2;
    lastv = m;
    b_i = iv0 + m;
    while ((lastv > 0) && (C[b_i - 2] == 0.0F)) {
      lastv--;
      b_i--;
    }

    lastc = n - 1;
    exitg2 = false;
    while ((!exitg2) && (lastc + 1 > 0)) {
      int exitg1;
      b_i = ic0 + lastc * 9;
      ia = b_i;
      do {
        exitg1 = 0;
        if (ia <= (b_i + lastv) - 1) {
          if (C[ia - 1] != 0.0F) {
            exitg1 = 1;
          } else {
            ia++;
          }
        } else {
          lastc--;
          exitg1 = 2;
        }
      } while (exitg1 == 0);

      if (exitg1 == 1) {
        exitg2 = true;
      }
    }
  } else {
    lastv = 0;
    lastc = -1;
  }

  if (lastv > 0) {
    float b_c;
    int c_i;
    if (lastc + 1 != 0) {
      if (lastc >= 0) {
        (void)memset(&work[0], 0,
                     (unsigned int)(int)(lastc + 1) * sizeof(float));
      }

      c_i = ic0 + 9 * lastc;
      for (iac = ic0; iac <= c_i; iac += 9) {
        b_c = 0.0F;
        b_i = (iac + lastv) - 1;
        for (ia = iac; ia <= b_i; ia++) {
          b_c += C[ia - 1] * C[((iv0 + ia) - iac) - 1];
        }

        b_i = div_nde_s32_floor(iac - ic0);
        work[b_i] += b_c;
      }
    }

    if (-tau != 0.0F) {
      b_i = ic0;
      for (iac = 0; iac <= lastc; iac++) {
        b_c = work[iac];
        if (b_c != 0.0F) {
          b_c *= -tau;
          c_i = lastv + b_i;
          for (ia = b_i; ia < c_i; ia++) {
            C[ia - 1] += C[((iv0 + ia) - b_i) - 1] * b_c;
          }
        }

        b_i += 9;
      }
    }
  }
}

/*
 * Arguments    : int n
 *                float *alpha1
 *                float x[81]
 *                int ix0
 * Return Type  : float
 */
__device__ float xzlarfg(int n, float *alpha1, float x[81], int ix0) {
  float tau;
  int k;
  tau = 0.0F;
  if (n > 0) {
    float xnorm;
    xnorm = xnrm2(n - 1, x, ix0);
    if (xnorm != 0.0F) {
      float beta1;
      beta1 = rt_hypotf(*alpha1, xnorm);
      if (*alpha1 >= 0.0F) {
        beta1 = -beta1;
      }

      if (fabsf(beta1) < 9.86076132E-32F) {
        int b_i;
        int knt;
        knt = 0;
        b_i = (ix0 + n) - 2;
        do {
          knt++;
          for (k = ix0; k <= b_i; k++) {
            x[k - 1] *= 1.01412048E+31F;
          }

          beta1 *= 1.01412048E+31F;
          *alpha1 *= 1.01412048E+31F;
        } while ((fabsf(beta1) < 9.86076132E-32F) && (knt < 20));

        beta1 = rt_hypotf(*alpha1, xnrm2(n - 1, x, ix0));
        if (*alpha1 >= 0.0F) {
          beta1 = -beta1;
        }

        tau = (beta1 - *alpha1) / beta1;
        xnorm = 1.0F / (*alpha1 - beta1);
        for (k = ix0; k <= b_i; k++) {
          x[k - 1] *= xnorm;
        }

        for (k = 0; k < knt; k++) {
          beta1 *= 9.86076132E-32F;
        }

        *alpha1 = beta1;
      } else {
        int b_i;
        tau = (beta1 - *alpha1) / beta1;
        xnorm = 1.0F / (*alpha1 - beta1);
        b_i = (ix0 + n) - 2;
        for (k = ix0; k <= b_i; k++) {
          x[k - 1] *= xnorm;
        }

        *alpha1 = beta1;
      }
    }
  }

  return tau;
}

/*
 * UNTITLED Summary of this function goes here
 *    Detailed explanation goes here
 *
 * Arguments    : const float x0[4]
 *                const float lb[4]
 *                const float ub[4]
 *                const float mmvek[10]
 *                const float b_fmod[5]
 *                const bool valid_freq[5]
 *                float sol[4]
 *                float *fval
 *                float *eflag
 *                struct0_T *output
 * Return Type  : void
 */
__device__ void SqpCuda(const float x0[4], const float lb[4], const float ub[4],
                        const float mmvek[10], const float b_fmod[5],
                        const bool valid_freq[5], float sol[4], float *fval,
                        float *eflag, struct0_T *output,
                        float optimality_tolerance) {
  b_struct_T memspace;
  c_struct_T QRManager;
  d_struct_T TrialState;
  e_struct_T WorkingSet;
  f_struct_T FiniteDifferences;
  i_coder_internal_stickyStruct FcnEvaluator;
  i_struct_T MeritFunction;
  j_struct_T QPObjective;
  k_struct_T CholManager;
  float absxk;
  float scale;
  float t;
  float t4_constrviolation;
  float t4_funcCount;
  float t4_iterations;
  int b_i;
  int i2;
  signed char WorkingSet_tmp[5];
  signed char obj_tmp_tmp[5];
  bool exitg1;
  for (b_i = 0; b_i < 5; b_i++) {
    FcnEvaluator.next.next.next.next.next.next.next.next.value.workspace
        .pi4fvek_c[b_i] = 12.566371F * b_fmod[b_i] / 3.0E+8F;
  }

  sol[0] = x0[0];
  sol[1] = x0[1];
  sol[2] = x0[2];
  sol[3] = x0[3];
  *eflag = 3.402823466E+38F;
  b_i = 0;
  exitg1 = false;
  while ((!exitg1) && (b_i < 4)) {
    if (lb[b_i] > ub[b_i]) {
      *eflag = -2.0F;
      exitg1 = true;
    } else {
      b_i++;
    }
  }

  if (*eflag == -2.0F) {
    *fval = 3.402823466E+38F;
    t4_iterations = 0.0F;
    t4_funcCount = 0.0F;
    output->algorithm[0] = 's';
    output->algorithm[1] = 'q';
    output->algorithm[2] = 'p';
    t4_constrviolation = 3.402823466E+38F;
    scale = 3.402823466E+38F;
    absxk = 3.402823466E+38F;
    t = 3.402823466E+38F;
  } else {
    float y;
    int mFixed;
    int mLB;
    int mUB;
    bool b;
    TrialState.nVarMax = 5;
    TrialState.mNonlinIneq = 0;
    TrialState.mNonlinEq = 0;
    TrialState.mIneq = 0;
    TrialState.mEq = 0;
    TrialState.iNonIneq0 = 1;
    TrialState.iNonEq0 = 1;
    TrialState.sqpFval_old = 0.0F;
    TrialState.sqpIterations = 0;
    TrialState.sqpExitFlag = 0;
    for (b_i = 0; b_i < 9; b_i++) {
      TrialState.lambdasqp[b_i] = 0.0F;
    }

    TrialState.steplength = 1.0F;
    for (b_i = 0; b_i < 5; b_i++) {
      TrialState.delta_x[b_i] = 0.0F;
    }

    TrialState.fstar = 0.0F;
    TrialState.firstorderopt = 0.0F;
    for (b_i = 0; b_i < 9; b_i++) {
      TrialState.lambda[b_i] = 0.0F;
    }

    TrialState.state = 0;
    TrialState.maxConstr = 0.0F;
    TrialState.iterations = 0;
    TrialState.xstarsqp[0] = x0[0];
    TrialState.xstarsqp[1] = x0[1];
    TrialState.xstarsqp[2] = x0[2];
    TrialState.xstarsqp[3] = x0[3];
    for (b_i = 0; b_i < 10; b_i++) {
      FcnEvaluator.next.next.next.next.next.next.next.next.value.workspace
          .mmvek[b_i] = mmvek[b_i];
    }

    for (b_i = 0; b_i < 5; b_i++) {
      FcnEvaluator.next.next.next.next.next.next.next.next.value.workspace
          .valid_freq[b_i] = valid_freq[b_i];
      FiniteDifferences.objfun.workspace.pi4fvek_c[b_i] =
          FcnEvaluator.next.next.next.next.next.next.next.next.value.workspace
              .pi4fvek_c[b_i];
    }

    for (b_i = 0; b_i < 10; b_i++) {
      FiniteDifferences.objfun.workspace.mmvek[b_i] = mmvek[b_i];
    }

    for (b_i = 0; b_i < 5; b_i++) {
      FiniteDifferences.objfun.workspace.valid_freq[b_i] = valid_freq[b_i];
    }

    FiniteDifferences.f_1 = 0.0F;
    FiniteDifferences.f_2 = 0.0F;
    FiniteDifferences.nVar = 4;
    FiniteDifferences.numEvals = 0;
    FiniteDifferences.SpecifyObjectiveGradient = false;
    FiniteDifferences.FiniteDifferenceType = 0;
    b = false;
    b_i = 0;
    while ((!b) && (b_i + 1 <= 4)) {
      FiniteDifferences.hasLB[b_i] = (lb[b_i] > -3.402823466E+38F);
      FiniteDifferences.hasUB[b_i] = (ub[b_i] < 3.402823466E+38F);
      if (FiniteDifferences.hasLB[b_i] || FiniteDifferences.hasUB[b_i]) {
        b = true;
      }

      b_i++;
    }

    while (b_i + 1 <= 4) {
      FiniteDifferences.hasLB[b_i] = (lb[b_i] > -3.402823466E+38F);
      FiniteDifferences.hasUB[b_i] = (ub[b_i] < 3.402823466E+38F);
      b_i++;
    }

    WorkingSet.nVar = 4;
    WorkingSet.nVarOrig = 4;
    WorkingSet.nVarMax = 5;
    WorkingSet.ldA = 5;
    for (b_i = 0; b_i < 5; b_i++) {
      WorkingSet.lb[b_i] = 0.0F;
      WorkingSet.ub[b_i] = 0.0F;
    }

    WorkingSet.mEqRemoved = 0;
    (void)memset(&WorkingSet.ATwset[0], 0, 45U * sizeof(float));
    WorkingSet.nActiveConstr = 0;
    for (b_i = 0; b_i < 9; b_i++) {
      WorkingSet.bwset[b_i] = 0.0F;
      WorkingSet.maxConstrWorkspace[b_i] = 0.0F;
      WorkingSet.isActiveConstr[b_i] = false;
      WorkingSet.Wid[b_i] = 0;
      WorkingSet.Wlocalidx[b_i] = 0;
    }

    WorkingSet.probType = 3;
    WorkingSet.SLACK0 = 1.0E-5;
    for (b_i = 0; b_i < 5; b_i++) {
      WorkingSet.nWConstr[b_i] = 0;
      WorkingSet.indexLB[b_i] = 0;
      WorkingSet.indexUB[b_i] = 0;
      WorkingSet.indexFixed[b_i] = 0;
    }

    mLB = 0;
    mUB = 0;
    mFixed = 0;
    for (b_i = 0; b_i < 4; b_i++) {
      bool guard1;
      scale = lb[b_i];
      guard1 = false;
      if (scale > -3.402823466E+38F) {
        if (fabsf(scale - ub[b_i]) < 0.001F) {
          mFixed++;
          WorkingSet.indexFixed[mFixed - 1] = b_i + 1;
        } else {
          mLB++;
          WorkingSet.indexLB[mLB - 1] = b_i + 1;
          guard1 = true;
        }
      } else {
        guard1 = true;
      }

      if (guard1 && (ub[b_i] < 3.402823466E+38F)) {
        mUB++;
        WorkingSet.indexUB[mUB - 1] = b_i + 1;
      }
    }

    b_i = (mLB + mUB) + mFixed;
    WorkingSet.mConstr = b_i;
    WorkingSet.mConstrOrig = b_i;
    WorkingSet.mConstrMax = 9;
    obj_tmp_tmp[0] = (signed char)mFixed;
    obj_tmp_tmp[1] = 0;
    obj_tmp_tmp[2] = 0;
    obj_tmp_tmp[3] = (signed char)mLB;
    obj_tmp_tmp[4] = (signed char)mUB;
    WorkingSet_tmp[0] = (signed char)mFixed;
    WorkingSet_tmp[1] = 0;
    WorkingSet_tmp[2] = 0;
    WorkingSet_tmp[3] = (signed char)(mLB + 1);
    WorkingSet_tmp[4] = (signed char)mUB;
    for (b_i = 0; b_i < 5; b_i++) {
      signed char c_i;
      signed char i1;
      c_i = obj_tmp_tmp[b_i];
      WorkingSet.sizes[b_i] = (int)c_i;
      WorkingSet.sizesNormal[b_i] = (int)c_i;
      i1 = WorkingSet_tmp[b_i];
      WorkingSet.sizesPhaseOne[b_i] = (int)i1;
      WorkingSet.sizesRegularized[b_i] = (int)c_i;
      WorkingSet.sizesRegPhaseOne[b_i] = (int)i1;
    }

    WorkingSet.isActiveIdxRegularized[0] = 1;
    WorkingSet.isActiveIdxRegularized[1] = mFixed;
    WorkingSet.isActiveIdxRegularized[2] = 0;
    WorkingSet.isActiveIdxRegularized[3] = 0;
    WorkingSet.isActiveIdxRegularized[4] = mLB;
    WorkingSet.isActiveIdxRegularized[5] = mUB;
    for (i2 = 0; i2 < 6; i2++) {
      WorkingSet.isActiveIdxRegPhaseOne[i2] =
          WorkingSet.isActiveIdxRegularized[i2];
    }

    for (b_i = 0; b_i < 5; b_i++) {
      WorkingSet.isActiveIdxRegPhaseOne[b_i + 1] +=
          WorkingSet.isActiveIdxRegPhaseOne[b_i];
    }

    for (i2 = 0; i2 < 6; i2++) {
      WorkingSet.isActiveIdx[i2] = WorkingSet.isActiveIdxRegPhaseOne[i2];
      WorkingSet.isActiveIdxNormal[i2] = WorkingSet.isActiveIdxRegPhaseOne[i2];
    }

    WorkingSet.isActiveIdxRegPhaseOne[0] = 1;
    WorkingSet.isActiveIdxRegPhaseOne[1] = mFixed;
    WorkingSet.isActiveIdxRegPhaseOne[2] = 0;
    WorkingSet.isActiveIdxRegPhaseOne[3] = 0;
    WorkingSet.isActiveIdxRegPhaseOne[4] = mLB + 1;
    WorkingSet.isActiveIdxRegPhaseOne[5] = mUB;
    for (i2 = 0; i2 < 6; i2++) {
      WorkingSet.isActiveIdxPhaseOne[i2] =
          WorkingSet.isActiveIdxRegPhaseOne[i2];
    }

    for (b_i = 0; b_i < 5; b_i++) {
      WorkingSet.isActiveIdxPhaseOne[b_i + 1] +=
          WorkingSet.isActiveIdxPhaseOne[b_i];
      WorkingSet.isActiveIdxRegularized[b_i + 1] +=
          WorkingSet.isActiveIdxRegularized[b_i];
      WorkingSet.isActiveIdxRegPhaseOne[b_i + 1] +=
          WorkingSet.isActiveIdxRegPhaseOne[b_i];
    }

    for (b_i = 0; b_i < mLB; b_i++) {
      i2 = WorkingSet.indexLB[b_i];
      TrialState.xstarsqp[i2 - 1] =
          fmaxf(TrialState.xstarsqp[i2 - 1], lb[i2 - 1]);
    }

    for (b_i = 0; b_i < mUB; b_i++) {
      i2 = WorkingSet.indexUB[b_i];
      TrialState.xstarsqp[i2 - 1] =
          fminf(TrialState.xstarsqp[i2 - 1], ub[i2 - 1]);
    }

    for (b_i = 0; b_i < mFixed; b_i++) {
      i2 = WorkingSet.indexFixed[b_i];
      TrialState.xstarsqp[i2 - 1] = ub[i2 - 1];
    }

    TrialState.sqpFval =
        sqp_cuda_anonFcn1(FcnEvaluator.next.next.next.next.next.next.next.next
                              .value.workspace.pi4fvek_c,
                          mmvek, valid_freq, TrialState.xstarsqp);
    (void)computeFiniteDifferences(&FiniteDifferences, TrialState.sqpFval,
                                   TrialState.xstarsqp, TrialState.grad, lb,
                                   ub);
    TrialState.FunctionEvaluations = FiniteDifferences.numEvals + 1;
    for (b_i = 0; b_i < mLB; b_i++) {
      WorkingSet.lb[WorkingSet.indexLB[b_i] - 1] =
          -lb[WorkingSet.indexLB[b_i] - 1] + x0[WorkingSet.indexLB[b_i] - 1];
    }

    for (b_i = 0; b_i < mUB; b_i++) {
      WorkingSet.ub[WorkingSet.indexUB[b_i] - 1] =
          ub[WorkingSet.indexUB[b_i] - 1] - x0[WorkingSet.indexUB[b_i] - 1];
    }

    for (b_i = 0; b_i < mFixed; b_i++) {
      scale = ub[WorkingSet.indexFixed[b_i] - 1] -
              x0[WorkingSet.indexFixed[b_i] - 1];
      WorkingSet.ub[WorkingSet.indexFixed[b_i] - 1] = scale;
      WorkingSet.bwset[b_i] = scale;
    }

    float Hessian[16];
    initActiveSet(&WorkingSet);
    MeritFunction.initFval = TrialState.sqpFval;
    MeritFunction.penaltyParam = 1.0F;
    MeritFunction.threshold = 0.0001F;
    MeritFunction.nPenaltyDecreases = 0;
    MeritFunction.linearizedConstrViol = 0.0F;
    MeritFunction.initConstrViolationEq = 0.0F;
    MeritFunction.initConstrViolationIneq = 0.0F;
    MeritFunction.phi = 0.0F;
    MeritFunction.phiPrimePlus = 0.0F;
    MeritFunction.phiFullStep = 0.0F;
    MeritFunction.feasRelativeFactor = 0.0F;
    MeritFunction.nlpPrimalFeasError = 0.0F;
    MeritFunction.nlpDualFeasError = 0.0F;
    MeritFunction.nlpComplError = 0.0F;
    MeritFunction.b_firstOrderOpt = 0.0F;
    MeritFunction.hasObjective = true;
    b_driver(lb, ub, &TrialState, &MeritFunction, &FcnEvaluator,
             &FiniteDifferences, &memspace, &WorkingSet, Hessian, &QRManager,
             &CholManager, &QPObjective, optimality_tolerance);
    sol[0] = TrialState.xstarsqp[0];
    sol[1] = TrialState.xstarsqp[1];
    sol[2] = TrialState.xstarsqp[2];
    sol[3] = TrialState.xstarsqp[3];
    *fval = TrialState.sqpFval;
    *eflag = (float)TrialState.sqpExitFlag;
    t4_iterations = (float)TrialState.sqpIterations;
    t4_funcCount = (float)TrialState.FunctionEvaluations;
    output->algorithm[0] = 's';
    output->algorithm[1] = 'q';
    output->algorithm[2] = 'p';
    t4_constrviolation = MeritFunction.nlpPrimalFeasError;
    scale = 1.29246971E-26F;
    absxk = fabsf(TrialState.delta_x[0]);
    if (absxk > 1.29246971E-26F) {
      y = 1.0F;
      scale = absxk;
    } else {
      t = absxk / 1.29246971E-26F;
      y = t * t;
    }

    absxk = fabsf(TrialState.delta_x[1]);
    if (absxk > scale) {
      t = scale / absxk;
      y = y * t * t + 1.0F;
      scale = absxk;
    } else {
      t = absxk / scale;
      y += t * t;
    }

    absxk = fabsf(TrialState.delta_x[2]);
    if (absxk > scale) {
      t = scale / absxk;
      y = y * t * t + 1.0F;
      scale = absxk;
    } else {
      t = absxk / scale;
      y += t * t;
    }

    absxk = fabsf(TrialState.delta_x[3]);
    if (absxk > scale) {
      t = scale / absxk;
      y = y * t * t + 1.0F;
      scale = absxk;
    } else {
      t = absxk / scale;
      y += t * t;
    }

    scale *= sqrtf(y);
    absxk = TrialState.steplength;
    t = MeritFunction.b_firstOrderOpt;
  }

  output->firstorderopt = t;
  output->lssteplength = absxk;
  output->stepsize = scale;
  output->constrviolation = t4_constrviolation;
  output->funcCount = t4_funcCount;
  output->iterations = t4_iterations;
}

/*
 * File trailer for sqp_cuda.c
 *
 * [EOF]
 */
