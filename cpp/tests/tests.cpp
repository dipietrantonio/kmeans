#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <exception>
#include <chrono>
#include <cstdlib>
#include "../include/kmeans.hpp"
#include <sstream>
#include <sched.h>

void test_simple_one_cluster_one_dimension(){
    std::vector<Point<1u>> dataset;
    double init_values[] {1.0f, 3.f, 2.f};
    dataset.push_back(Point<1u>(reinterpret_cast<double*>(init_values)));
    dataset.push_back(Point<1u>(reinterpret_cast<double*>(init_values) + 1));
   
    std::vector<Point<1u>> expected_centres {Point<1u>(reinterpret_cast<double*>(init_values) + 2)};
    std::vector<Point<1u>> computed_centres;
    std::vector<int> assignment;
    std::tie(computed_centres, assignment) = kmeans(dataset, 1u);
    if(computed_centres != expected_centres) throw std::exception();
    std::vector<int> expected_assignment {0, 0};
    std::cout << "Test 'test_simple_one_cluster_one_dimension' passed." << std::endl;
}


void test_scatter_dataset(){
    char *dsname {std::getenv("KMEANS_DATASET")};
    if(!dsname){
        std::cerr << "Dataset file path not specified through the KMEANS_DATASET environment variable." << std::endl;
        throw std::exception();
    }
    
    int world_size, world_rank;
    MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    std::vector<Point<2u>> dataset;
    if(world_rank == 0)
    read_dataset(dataset, dsname);
    std::vector<Point<2u>> local_dataset;
    size_t n;
    scatter_dataset(dataset, local_dataset, n);
    std::stringstream ss;
    //ss << "Rank " << world_rank << ", ds size: " << local_dataset.size() << ", elements: ";
    //for(int i {0}; i < local_dataset.size(); i++) ss << local_dataset[i] << ", ";
    //ss << std::endl;
    // std::cout << ss.str() ;
}


void test_corner_case_1(){
    std::vector<Point<2u>> dataset;
    double init_values[] {1.0f, 1.0f, 2.f, 2.f};
    dataset.push_back(Point<2u>(reinterpret_cast<double*>(init_values)));
    dataset.push_back(Point<2u>(reinterpret_cast<double*>(init_values) + 2));
   
    std::vector<Point<2>> expected_centres;
    expected_centres.push_back(Point<2u>(reinterpret_cast<double*>(init_values)));
    expected_centres.push_back(Point<2u>(reinterpret_cast<double*>(init_values) + 2));
    std::vector<Point<2u>> computed_centres;
    std::vector<int> assignment;
    std::tie(computed_centres, assignment) = kmeans(dataset, 2u);
    if(computed_centres != expected_centres) throw std::exception();
    std::cout << "Test 'test_corner_case_1' passed." << std::endl;
}



void test_with_dataset(){
    char *dsname {std::getenv("KMEANS_DATASET")};
    if(!dsname){
        std::cerr << "Dataset file path not specified through the KMEANS_DATASET environment variable." << std::endl;
        throw std::exception();
    }
    
    std::vector<Point<2u>> dataset;
    int world_rank;
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    
    int cpu = sched_getcpu();
    std::cout << "Rank " << world_rank << " is running on CPU " << cpu <<std::endl;
    if(world_rank == 0)
        read_dataset(dataset, dsname);
    std::vector<Point<2u>> centres;
    std::vector<int> assignment;
    //auto start = std::chrono::steady_clock::now();
    std::tie(centres, assignment) = kmeans(dataset, 2);
    //auto end = std::chrono::steady_clock::now();
    //std::cout << "Kmeans execution time: " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << std::endl;
    std::cout << "Center 0: " << centres[0] << ", centre 1: " << centres[1] << std::endl; 
}



int main(void){
    MPI_CHECK_ERROR(MPI_Init(NULL, NULL));
    //test_scatter_dataset();
    //test_simple_one_cluster_one_dimension();
    //test_corner_case_1();
     test_with_dataset();
    MPI_CHECK_ERROR(MPI_Finalize());
    std::cout << "All tests passed." << std::endl;
}
