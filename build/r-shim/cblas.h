#pragma once

/* Minimal CBLAS definitions needed by MLX (GEMM). */

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

void cblas_sgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void cblas_dgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

void cblas_cgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

void cblas_zgemm(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
