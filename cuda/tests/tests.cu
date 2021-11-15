#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <exception>
#include "../include/kmeans.hpp"
#include <chrono>

void test_simple_one_cluster_one_dimension(){
    std::vector<Point<1u>> dataset;
    float init_values[] {1.0f, 3.f, 2.f};
    dataset.push_back(Point<1u>(reinterpret_cast<float*>(init_values)));
    dataset.push_back(Point<1u>(reinterpret_cast<float*>(init_values) + 1));
   
    std::vector<Point<1u>> expected_centres {Point<1u>(reinterpret_cast<float*>(init_values) + 2)};
    std::vector<Point<1u>> computed_centres;
    std::vector<int> assignment;
    std::tie(computed_centres, assignment) = kmeans(dataset, 1u);
    std::vector<int> expected_assignment {0, 0};
    if(computed_centres != expected_centres) throw std::exception();
    std::cerr << "Test 'test_simple_one_cluster_one_dimension' passed." << std::endl;
}



void test_corner_case_1(){
    std::vector<Point<2u>> dataset;
    float init_values[] {1.0f, 1.0f, 2.f, 2.f};
    dataset.push_back(Point<2u>(reinterpret_cast<float*>(init_values)));
    dataset.push_back(Point<2u>(reinterpret_cast<float*>(init_values) + 2));
   
    std::vector<Point<2>> expected_centres;
    expected_centres.push_back(Point<2u>(reinterpret_cast<float*>(init_values)));
    expected_centres.push_back(Point<2u>(reinterpret_cast<float*>(init_values) + 2));
    std::vector<Point<2u>> computed_centres;
    std::vector<int> assignment;
    std::tie(computed_centres, assignment) = kmeans(dataset, 2u);
    if(computed_centres != expected_centres) throw std::exception();
    std::vector<int> expected_assignment {0, 0};
    std::cerr << "Test 'test_corner_case_1' passed." << std::endl;
}



void test_with_dataset(){
    std::vector<Point<2u>> dataset;
    read_dataset(dataset, "tests/dataset.txt");
    float epval[] {-0.08349970350381591f, -0.08535355221426252f, 1.0730989137683584f, 1.0762199531299013f};
    std::vector<Point<2>> centres;
    std::vector<int> assignment;
    auto start = std::chrono::steady_clock::now();
    std::tie(centres, assignment) = kmeans(dataset, 2);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Kmeans execution time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
    if(centres[0].distance(Point<2u>(epval)) >= 1e-5) throw std::exception();
    if(centres[1].distance(Point<2u>(epval+2)) >= 1e-5) throw std::exception();
}



int main(void){
    test_simple_one_cluster_one_dimension();
    test_corner_case_1();
    test_with_dataset();
    std::cout << "All tests passed." << std::endl;
}