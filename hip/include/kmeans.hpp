#include <hip/hip_runtime.h>
#ifndef __KMEANS_HPP__
#define  __KMEANS_HPP__

#include <tuple>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <iostream>
#include <fstream>
#include <cstring>

#define WARP_SIZE 64 
#define NTHREADS 1024
#define NWARPS (NTHREADS / WARP_SIZE)

#define CUDA_CHECK_ERROR(X)({\
    if((X) != hipSuccess){\
        std::cerr << "HIP ERROR " << (X) << ": " << hipGetErrorString((X)) << "\n";\
        exit(1);\
    }\
})



template <unsigned int dim>
class Point {
    
    public:
    float coord[dim];


    __host__ __device__
    Point(){
    };

    __host__ __device__
	void zero(){
        memset(coord, 0, sizeof(float) * dim);
    };
    
    __host__ __device__
    Point(float *coord){
        for(unsigned int i {0u}; i < dim; i++)
            this->coord[i] = coord[i];
    };


    bool operator==(const Point<dim>& other) const {
        for (unsigned int i {0u}; i < dim; i++){
            if(std::abs(this->coord[i] - other.coord[i]) > std::numeric_limits<float>::epsilon()){
                return false;
            }
        }
        return true;
    };


    friend std::ostream& operator<<(std::ostream& os, const Point<dim>& p){
        os << "(";
        for(unsigned int d {0}; d < dim - 1; d++)
            os << p.coord[d] << ", ";
        os << p.coord[dim-1] << ")";
        return os; 
    }

    __host__ __device__
    void operator+=(const Point<dim>& other){
        for(unsigned int i {0u}; i < dim; i++){
            coord[i] += other.coord[i];
        }
    }


    template <typename T>
    __host__ __device__
    void operator/=(const T& other){
        for(unsigned int i {0u}; i < dim; i++){
            coord[i] /= other;
        }
    }

    __host__ __device__
    float distance(const Point<dim> &other) const {
        float dist  {0.0f};
        for (unsigned int i {0u}; i < dim; i++)
            dist += powf(this->coord[i] - other.coord[i], 2);
        return sqrt(dist);
    };
};



__device__ int globally_converged;



template <unsigned int dim> 
__global__ void kmeans_compute_assignment(Point<dim> *points, size_t n_ponts, Point<dim> *centres, size_t K, int *assignment){
    unsigned int idx_start {threadIdx.x + blockIdx.x * blockDim.x};
    unsigned int warp_id {threadIdx.x / WARP_SIZE};
    unsigned int lane_id {threadIdx.x % WARP_SIZE};
    unsigned int grid_size {blockDim.x * gridDim.x};
    bool converged {true};
    __shared__ bool block_converged[NWARPS - 1];
    // step one: assign each point to a center
    for(unsigned int idx {idx_start}; idx < n_ponts; idx += grid_size){
        float best_dist {points[idx].distance(centres[0])};
        unsigned int chosen_centre {0u};
        for(unsigned int i {1u}; i < K; i++){
            float current_dist {points[idx].distance(centres[i])};
            if(current_dist < best_dist){
                best_dist = current_dist;
                chosen_centre = i;
            }
        }
        if(assignment[idx] != chosen_centre){
            assignment[idx] = chosen_centre;
            converged = false;
        }
    }
    // step 2: determine if computation has converged.
    // ..within the warp
    converged = __all(converged);
    // ..within the block
    if(warp_id > 0 && lane_id == 0) block_converged[warp_id - 1] = converged;
    __syncthreads();
    if(warp_id == 0){
        if(lane_id > 0){
		    if(lane_id < NWARPS) converged = block_converged[lane_id-1];
		    else converged = true;
	    }
        converged = __all(converged);
        // within all blocks
        if(lane_id == 0){
            int int_converged {converged ? 1 : 0};
            atomicCAS(&globally_converged, 1, int_converged);
        }
    }
}



template <unsigned int dim> 
__global__ void kmeans_compute_centres(Point<dim> *points, size_t n_ponts, Point<dim> *centres, size_t K, int *assignment, volatile int *counters, volatile int *mutex, volatile int *block_counters){
    unsigned int idx_start {threadIdx.x + blockIdx.x * blockDim.x};
    unsigned int warp_id {threadIdx.x / WARP_SIZE};
    unsigned int lane_id {threadIdx.x % WARP_SIZE};
    unsigned int grid_size {blockDim.x * gridDim.x};
    __shared__ Point<dim> shared_centres[NWARPS - 1];
    __shared__ size_t shared_counters[NWARPS - 1];

    for(unsigned int i {0u}; i < K; i++){
        Point<dim> local_centre {};
        local_centre.zero();
        unsigned int counter {0u};
        for(unsigned int idx {idx_start}; idx < n_ponts; idx += grid_size){
            if(assignment[idx] == i){
                local_centre += points[idx];
                counter++;
            }
        }
        // warp reduction
        for(unsigned int o {WARP_SIZE/2}; o > 0; o >>= 1){
            for(unsigned int d {0u}; d < dim; d++){
                float comp = __shfl_down(local_centre.coord[d], o);
                if(lane_id < o){
                    local_centre.coord[d] += comp;
                }
            }
            unsigned int up = __shfl_down(counter, o);
            if(lane_id < o){
                counter += up;
            }
        }
        if(lane_id == 0 && warp_id > 0){
            shared_centres[warp_id - 1] = local_centre;
            shared_counters[warp_id - 1] = counter;
        }
        __syncthreads();
        if(warp_id == 0){
            if(lane_id > 0){
	        if(lane_id < NWARPS){
                	local_centre = shared_centres[lane_id - 1];
        	       	counter = shared_counters[lane_id - 1];
	        }else{
                    local_centre.zero();
                    counter = 0;
                }
            }
        
            for(unsigned int o {WARP_SIZE/2}; o > 0; o >>= 1){
                for(unsigned int d {0u}; d < dim; d++){
                    float comp = __shfl_down(local_centre.coord[d], o);
                    if(lane_id < o){
                        local_centre.coord[d] += comp;
                    }
                }
                unsigned int up = __shfl_down(counter, o);
                if(lane_id < o){
                    counter += up;
                }
            }
            // contribute to the final result
            if(lane_id == 0){
                // the following is ok because only one thread per warp executes this.
                // otherwise do not do it!
                while(atomicCAS((int*)mutex + i, 1, -1) != 1);
                __threadfence();
                centres[i] += local_centre;
                counters[i] += counter;
                if(block_counters[i] == gridDim.x - 1){
                    centres[i] /= counters[i];
                    block_counters[i] = 0;
                }else{
                    block_counters[i] += 1;
                }
                mutex[i] = 1;
            }
        }
    }
}



template <unsigned int dim> 
__global__ void init_arrays(int *assignment, int *mutex, Point<dim>* points, Point<dim>* centres, size_t n_points, unsigned int K){
    unsigned int idx_start {blockDim.x * blockIdx.x + threadIdx.x};
    unsigned int grid_size {blockDim.x * gridDim.x}; 
    for(unsigned int idx {idx_start}; idx < K; idx += grid_size){
        centres[idx] = points[idx];
        mutex[idx] = 1;
    }
    for(unsigned int idx {idx_start}; idx < n_points; idx += grid_size){
        assignment[idx] = -1;
    }
}



/**
 * Performs k-means clustering on a dataset using K centres.
 * @param dataset a vector of points.
 * @param K number of centres.
 * @returns A tuple where the first element is the vector of centres
 * and the second one is the assignment of points to centres.
 */
template <unsigned int dim>
std::tuple<std::vector<Point<dim>>, std::vector<int>> kmeans (std::vector<Point<dim>>& dataset, unsigned int K){
    if(dataset.size() < K){
        std::cerr << "Not enough points in the dataset for the required number of centres.\n";
        throw std::exception();
    }else if(K == 0){
        std::cerr << "Invalid K value (0)\n";
        throw std::exception();
    }

    // allocate memory on GPU
    Point<dim> *dev_points, *dev_centres;
    int *dev_assignment, *dev_counters, *dev_mutex, *dev_block_counters;
    CUDA_CHECK_ERROR(hipMalloc(&dev_points, sizeof(Point<dim>) * dataset.size()));
    CUDA_CHECK_ERROR(hipMalloc(&dev_centres, sizeof(Point<dim>) * K));
    CUDA_CHECK_ERROR(hipMalloc(&dev_assignment, sizeof(int) * dataset.size()));
    CUDA_CHECK_ERROR(hipMalloc(&dev_counters, sizeof(int) * K));
    CUDA_CHECK_ERROR(hipMalloc(&dev_mutex, sizeof(int) * K));
    CUDA_CHECK_ERROR(hipMalloc(&dev_block_counters, sizeof(int) * K));
    CUDA_CHECK_ERROR(hipMemset(dev_block_counters, 0, sizeof(int) * K));
    CUDA_CHECK_ERROR(hipMemcpy(dev_points, dataset.data(), sizeof(Point<dim>) * dataset.size(), hipMemcpyHostToDevice));
    // call kernel to init assignment and centres.
    struct hipDeviceProp_t props;
    CUDA_CHECK_ERROR(hipGetDeviceProperties(&props, 0));
    const unsigned int nblocks {2u * props.multiProcessorCount};
    hipLaunchKernelGGL(init_arrays, dim3(nblocks), dim3(NTHREADS), 0, 0, dev_assignment, dev_mutex, dev_points, dev_centres, dataset.size(), K);
    CUDA_CHECK_ERROR(hipGetLastError());
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    const int converged_init {1};
    CUDA_CHECK_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(globally_converged), &converged_init, sizeof(int)));
    int has_converged;
    Point<dim> *tmp_centres = new Point<dim>[K];
    int iter {0};
    while(true){
        hipLaunchKernelGGL(kmeans_compute_assignment, dim3(nblocks), dim3(NTHREADS), 0, 0, dev_points, dataset.size(), dev_centres, K, dev_assignment);
        CUDA_CHECK_ERROR(hipGetLastError());
        CUDA_CHECK_ERROR(hipMemcpy(tmp_centres, dev_centres, sizeof(Point<dim>) * K, hipMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(hipDeviceSynchronize());
        iter++;
        CUDA_CHECK_ERROR(hipMemcpyFromSymbol(&has_converged, HIP_SYMBOL(globally_converged), sizeof(int)));
        if(has_converged) break;
        CUDA_CHECK_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(globally_converged), &converged_init, sizeof(int)));
        CUDA_CHECK_ERROR(hipMemset(dev_counters, 0, sizeof(int) * K));
        CUDA_CHECK_ERROR(hipMemset(dev_centres, 0, sizeof(Point<dim>) * K));
        hipLaunchKernelGGL(kmeans_compute_centres, dim3(nblocks), dim3(NTHREADS), 0, 0, dev_points, dataset.size(), dev_centres, K, dev_assignment, dev_counters, dev_mutex, dev_block_counters);
    }
    // copy back assignment and centres to memory
    std::vector<int> assignment(dataset.size());
    std::vector<Point<dim>> centres(K);

    CUDA_CHECK_ERROR(hipMemcpy(assignment.data(), dev_assignment, sizeof(int) * assignment.size(), hipMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(hipMemcpy(centres.data(), dev_centres, sizeof(Point<dim>) * centres.size(), hipMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    return {centres, assignment};
}



template <unsigned int dim>
void read_dataset(std::vector<Point<dim>>& dataset, const char *filename){
    std::ifstream in {filename};
    unsigned int n_points;
    in >> n_points;
    float coord[dim];
    for(unsigned int i {0u}; i < n_points; i++){
        in >> coord[0];
        in >> coord[1];
        dataset.push_back(Point<dim>(coord));
    }
    in.close();
}

#endif
