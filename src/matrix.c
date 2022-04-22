#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/* Transposes the matrix */
void transpose(matrix *result, matrix *mat) {
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[j * mat->rows + i] = mat->data[i * mat->cols + j];
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int idx = row * mat->cols + col;
    return mat->data[idx];
}

/*
 * Sets the value at the given row and column to val.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int idx = row * mat->cols + col;
    mat->data[idx] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        return -1;
    }
    double* mat_data = calloc(rows*cols, sizeof(double));
    if (mat_data == NULL) {
        return -2;
    }
    matrix *new_mat = malloc(sizeof(matrix));
    if (new_mat == NULL) {
        return -2;
    }
    new_mat->rows = rows;
    new_mat->cols = cols;
    new_mat->data = mat_data;
    new_mat->parent = NULL;
    new_mat->ref_cnt = 1;
    *mat = new_mat;
    return 0;
}

/*
 * Deallocates matrix.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *new_mat = malloc(sizeof(matrix));
    if (new_mat == NULL) {
        return -2;
    }
    new_mat->data = from->data + offset;
    new_mat->rows = rows;
    new_mat->cols = cols;
    new_mat->parent = from;
    from->ref_cnt += 1;
    *mat = new_mat;
    return 0;
}

/*
 * Sets all entries in mat to val.
 */
void fill_matrix(matrix *mat, double val) {
    #pragma omp parallel for
    for (int i = 0; i < (mat->cols * mat->rows) / 4 * 4; i += 4) {
        _mm256_storeu_pd(mat->data + i, _mm256_set1_pd(val));
    }
    #pragma omp parallel for
    for (int i = (mat->cols * mat->rows) / 4 * 4; i < mat->cols * mat->rows; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    __m256d neg_mask = _mm256_set1_pd(-1);
    #pragma omp parallel for
    for (int i = 0; i < (mat->cols * mat->rows) / 4 * 4; i += 4) {
        __m256d tmp = _mm256_loadu_pd(mat->data + i);
        __m256d neg_tmp = _mm256_mul_pd(neg_mask, tmp);
        _mm256_storeu_pd(result->data + i, _mm256_max_pd(tmp, neg_tmp));
    }
    #pragma omp parallel for
    for (int i = (mat->cols * mat->rows) / 4 * 4; i < mat->cols * mat->rows; i++) {
        result->data[i] = abs(mat->data[i]);
    }
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    __m256d mat1_vec, mat2_vec;
    #pragma omp parallel for
    for (int i = 0; i < (mat1->cols * mat1->rows) / 4 * 4; i += 4) {
        mat1_vec = _mm256_loadu_pd(mat1->data + i);
        mat2_vec = _mm256_loadu_pd(mat2->data + i);
        _mm256_storeu_pd(result->data + i, _mm256_add_pd(mat1_vec, mat2_vec));
    }
    #pragma omp parallel for
    for (int i = (mat1->cols * mat1->rows) / 4 * 4; i < mat1->cols * mat1->rows; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    matrix *temp_result, *transposed;
    allocate_matrix(&temp_result, result->rows, result->cols);
    allocate_matrix(&transposed, mat2->cols, mat2->rows);
    transpose(transposed, mat2);
    #pragma omp parallel for
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < transposed->rows; j++) {
            __m256d sum_vec = _mm256_setzero_pd();
            for (int k = 0; k < transposed->cols / 32 * 32; k += 32) {
                __m256d mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k);
                __m256d mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 4);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 4);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 8);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 8);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 12);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 12);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 16);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 16);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 20);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 20);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 24);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 24);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
                mat1_vec = _mm256_loadu_pd(mat1->data + i * mat1->cols + k + 28);
                mat2_vec = _mm256_loadu_pd(transposed->data + j * transposed->cols + k + 28);
                sum_vec = _mm256_fmadd_pd(mat1_vec, mat2_vec, sum_vec);
            }
            double tmp_arr[4];
            _mm256_storeu_pd(tmp_arr, sum_vec);
            double sum_num = tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3];
            for (int k = transposed->cols / 32 * 32; k < transposed->cols; k++) {
                sum_num += mat1->data[i * mat1->cols + k] * transposed->data[j * transposed->cols + k];
            }
            int idx = i * temp_result->cols + j;
            temp_result->data[idx] = sum_num;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < result->rows*result->cols; i++) {
        result->data[i] = temp_result->data[i];
    }
    deallocate_matrix(temp_result);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (pow == 0) {
        #pragma omp parallel for
        for (int i = 0; i < result->rows; i++) {
            result->data[i * result->cols + i] = 1;
        }
        return 0;
    }
    #pragma omp parallel for
    for (int i = 0; i < result->rows * result->cols; i++) {
        result->data[i] = mat->data[i];
    }
    matrix *y;
    allocate_matrix(&y, result->rows, result->cols);
    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        y->data[i * result->cols + i] = 1;
    }
    while (pow > 1) {
        if (pow % 2 == 0) {
            mul_matrix(result, result, result);
            pow = pow / 2;
        } else {
            mul_matrix(y, result, y);
            mul_matrix(result, result, result);
            pow = (pow - 1) / 2;
        }
    }
    mul_matrix(result, result, y);
    deallocate_matrix(y);
    return 0;
}