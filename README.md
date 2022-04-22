
# Parallel Matrix Computation in C

### A MIMD performance floating point matrix computation library  written in C.

Accelerated by multiple-instruction, multiple-data stream to achieve >700x time optimization for matrix powering:

- Intel Streaming SIMD Extensions to pack values into vectorized registers (data-level parallelism)
	- Runs on x86 processors that support AVX2
	- 256-bit YMM registers packed with 64-bit floating point values (_mm256\_* intrinsics)
	- In order to vectorize dot product operations, transposed second matrix operand to leverage cache blocking and locality
- OpenMP compiler directives to allocate parallel threads in computation (thread-level parallelism)
	- AVX data-types declared in block-scope for implicit private thread variables
- Programmatically optimize CPU pipelining (instruction-level parallelism)
	- Load delay slots mitigated via grouping _mm256\_* intrinsic load instructions so no performance loss by stalling for subsequent ops which may depend on loaded value(s)
	- Unrolled loops to mitigate branch penalty and loop overhead in parallel computation

`void  transpose(matrix *result, matrix *mat)` 

Description 
- Store the result transposing `mat` element-wise to `result`.

Returns
- None.

`int  allocate_matrix(matrix **mat, int  rows, int  cols)` 

Description 
- Allocates space for a matrix struct pointed to by the double pointer mat with `rows` rows and `cols` columns.

Returns
- Returns 0 upon success.

`void  deallocate_matrix(matrix *mat)` 

Description 
- Frees `mat->data` if `mat` is not a slice and has no existing slices, frees `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references (including itself).

Returns
- None.

`int  allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols)` 

Description 
- Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.

Returns
- Returns -1 if either `rows` or `cols` or both have invalid values. Returns -2 if any call to allocate memory in this function fails. Returns 0 upon success.

`void  fill_matrix(matrix *mat, double val)` 

Description 
- Sets all entries in `mat` to `val`. Note that the matrix is in row-major order.

Returns
- None.

`int  abs_matrix(matrix *result, matrix *mat)` 

Description 
- Store the result of taking the absolute value element-wise to `result`.

Returns
- Returns 0 upon success.

`int  add_matrix(matrix *result, matrix *mat1, matrix *mat2)` 

Description 
- Store the result of adding `mat1` and `mat2` to `result`.

Returns
- Returns 0 upon success.

`int  mul_matrix(matrix *result, matrix *mat1, matrix *mat2)` 

Description 
- Store the result of multiplying `mat1` and `mat2` to `result`.

Returns
- Returns 0 upon success.

`int  pow_matrix(matrix *result, matrix *mat, int  pow)` 

Description 
- Store the result of raising `mat` to the (`pow`)th power to `result`.

Returns
- Returns 0 upon success.


### Benchmarks against naive matrix computation implementations:

| Test Suite                 | Dimensional Bounds (100 <= n <= 10000) | Speed-up Factor  (From Naive Implementation) | Correctness  Integration  Test Status |
|---------------------------|---------------------------------------|----------------------------------------------|---------------------------------------|
| Matrix Multiply           | 10000 x 10000                         | 60.078325                                    | PASS                                  |
| Matrix Powering           | 10000 x 10000                         | 731.851445                                   | PASS                                  |
| Simple Benchmark (Add, Abs)         | 800 x 800                             | 2.687932                                     | PASS                                  |
| Comprehensive Integration (Add, Abs) | 2400 x 2400                           | 48.307556                                    | PASS                                  |