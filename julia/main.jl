using CUDA
using BenchmarkTools

function matrix_multiply_hand!(A, x, result)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    n_rows = size(A, 1)
    n_cols = size(A, 2)
    for row in index:stride:n_rows
	sum = 0.0
	for col in 1:n_cols
	    @inbounds sum += A[row, col] * x[row]
	end
	@inbounds result[row] = sum
    end
    return nothing
end

function bench_matrix_multiply_hand!(A, x, result)
    num_threads = 512
    num_blocks = ceil(Int, size(A,1) / num_threads)
    CUDA.@sync begin
	@cuda threads=num_threads blocks=num_blocks matrix_multiply_hand!(A, x, result)
    end
end

function matrix_multiply_blas!(A, x, result)
    return A*x
end

N = 4096
M = 1024

A = CuArray(ones(N, M))
x = CuArray(ones(M))
result = CuArray(ones(M))

@btime bench_matrix_multiply_hand!(A, x, result)
@btime matrix_multiply_blas!(A, x, result)
