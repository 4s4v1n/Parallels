/**
 * @file    interface.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "interface.h"

#include <chrono>
#include <ctime>

namespace s21 {

AntInterface::AntInterface() : m_isCorrectGraphLoad(false) {}

int AntInterface::mainMenuItem() {
  std::cin.exceptions(std::ios_base::failbit);
  int action;
  std::cout << " __________________________________________________________\n"
            << "|                                                         |\n"
            << "|                     Сhoose action:                      |\n"
            << "|                                                         |\n"
            << "| 1. Load graph from file                                 |\n"
            << "| 2. Set number of iterations                             |\n"
            << "| 3. Research                                             |\n"
            << "|                                                         |\n"
            << "| 0. Exit                                                 |\n"
            << "|_________________________________________________________|\n"
            << std::endl
            << ">>> ";

  std::cin >> action;

  return action;
}

auto AntInterface::SetNumberOfIterations() -> void {
  int iters;
  std::cin >> iters;
  if (iters <= 0) {
    std::cout << "number of iterations must be positive\n";
  } else {
    std::cout << "\nsuccess!\n";
    m_number_of_iterations = iters;
  }
}

auto AntInterface::LoadGraph() -> void {
  try {
    std::cout << "\nEnter file path:" << std::endl << ">>> ";
    std::string path;
    std::cin >> path;
    m_graph.loadGraphFromFile(path);
    m_isCorrectGraphLoad = true;
    std::cout << "\nDownload successful!\n";
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

auto AntInterface::Research() -> void {
  if (!m_isCorrectGraphLoad) {
    std::cout << "\nMatrix not loaded\n";
    std::cout << "\nPress Enter to continue...\n";
    system("read REPLY");
    return;
  }
  auto startTime = std::chrono::steady_clock::now();
  for (int i = 0; i < m_number_of_iterations; i++) {
    GraphAlgorithms::solveTravelingSalesmanProblem(m_graph);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> commonAlgTime = end - startTime;

  startTime = std::chrono::steady_clock::now();
  for (int i = 0; i < m_number_of_iterations; i++) {
    GraphAlgorithms::solveTravelingSalesmanProblemParalell(m_graph);
  }
  end = std::chrono::steady_clock::now();
  std::chrono::duration<double> parallelAlgTime = end - startTime;

  std::cout << "\ncommon TSP algorithm time: " << commonAlgTime.count()
            << " sec\n";
  std::cout << "parallel TSP algorithm time: " << parallelAlgTime.count()
            << " sec\n";
}

auto AntInterface::Show() -> void {
  while (true) {
    try {
      system("clear");

      while (true) {
        std::cin.exceptions(std::ios_base::failbit);
        int action = mainMenuItem();
        switch (action) {
          case 1:
            LoadGraph();
            break;
          case 2:
            SetNumberOfIterations();
            break;
          case 3:
            Research();
            break;
          case 0:
            return;
          default:
            std::cout << "Choose the correct action" << std::endl;
            std::cin.clear();
            break;
        }
        system("read REPLY");
        getchar();
        system("clear");
      }

      system("clear");
    } catch (std::exception &e) {
      std::cout << "\n Check the correctness of the input" << std::endl;
      std::cin.clear();
      system("read REPLY");
      getchar();
      system("clear");
    }
  }
}
}  // namespace s21