#pragma once

/* Minimal LAPACK prototypes used by MLX. Names follow the Fortran
 * underscore convention and match OpenBLAS/Netlib C headers. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float  r; float  i; } lapack_complex_float;
typedef struct { double r; double i; } lapack_complex_double;

void sgeqrf_(const int *m, const int *n, float *a, const int *lda, float *tau,
             float *work, const int *lwork, int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau,
             double *work, const int *lwork, int *info);

void sorgqr_(const int *m, const int *n, const int *k, float *a, const int *lda,
             const float *tau, float *work, const int *lwork, int *info);
void dorgqr_(const int *m, const int *n, const int *k, double *a, const int *lda,
             const double *tau, double *work, const int *lwork, int *info);

void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a, const int *lda,
             float *w, float *work, const int *lwork, int *iwork, const int *liwork, int *info);
void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
             double *w, double *work, const int *lwork, int *iwork, const int *liwork, int *info);

void sgeev_(const char *jobvl, const char *jobvr, const int *n, float *a, const int *lda,
            float *wr, float *wi, float *vl, const int *ldvl, float *vr, const int *ldvr,
            float *work, const int *lwork, int *info);
void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda,
            double *wr, double *wi, double *vl, const int *ldvl, double *vr, const int *ldvr,
            double *work, const int *lwork, int *info);

void spotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);

void sgesdd_(const char *jobz, const int *m, const int *n, float *a, const int *lda,
             float *s, float *u, const int *ldu, float *vt, const int *ldvt,
             float *work, const int *lwork, int *iwork, int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a, const int *lda,
             double *s, double *u, const int *ldu, double *vt, const int *ldvt,
             double *work, const int *lwork, int *iwork, int *info);

void sgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);

void sgetri_(const int *n, float *a, const int *lda, const int *ipiv,
             float *work, const int *lwork, int *info);
void dgetri_(const int *n, double *a, const int *lda, const int *ipiv,
             double *work, const int *lwork, int *info);

void strtri_(const char *uplo, const char *diag, const int *n, float *a, const int *lda, int *info);
void dtrtri_(const char *uplo, const char *diag, const int *n, double *a, const int *lda, int *info);

void cheevd_(const char *jobz, const char *uplo, const int *n, lapack_complex_float *a, const int *lda,
             float *w, lapack_complex_float *work, const int *lwork,
             float *rwork, const int *lrwork, int *iwork, const int *liwork, int *info);
void zheevd_(const char *jobz, const char *uplo, const int *n, lapack_complex_double *a, const int *lda,
             double *w, lapack_complex_double *work, const int *lwork,
             double *rwork, const int *lrwork, int *iwork, const int *liwork, int *info);

#ifdef __cplusplus
}  // extern "C"
#endif
