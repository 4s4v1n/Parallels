/**
 * @file    gauss_interface.cpp
 * @authors telvina<at>student.21-school.ru
 * @authors sreanna<at>student.21-school.ru
 * @authors jmadie<at>student.21-school.ru
 * @date    13/09/22
 */

#include "gauss_interface.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace s21 {

int GaussInterface::mainMenuItem() {
  std::cin.exceptions(std::ios_base::failbit);
  int action;
  std::cout << " __________________________________________________________\n"
            << "|                                                         |\n"
            << "|            Solution of a system of linear               |\n"
            << "|       algebraic equations by the Gauss method           |\n"
            << "|                                                         |\n"
            << "|                     Сhoose action:                      |\n"
            << "|                                                         |\n"
            << "| 1. Load matrix from file                                |\n"
            << "| 2. Print matrix                                         |\n"
            << "| 3. Solve the equation without parallel                  |\n"
            << "| 4. Solve the equation with parallel                     |\n"
            << "| 5. Measure the execution time of an algorithm           |\n"
            << "|                                                         |\n"
            << "| 0. Exit                                                 |\n"
            << "|_________________________________________________________|\n"
            << std::endl
            << ">>> ";

  std::cin >> action;

  return action;
}

auto GaussInterface::loadMatrix() -> void {
  try {
    std::cout << "\nEnter file path:" << std::endl << ">>> ";
    std::string path;
    std::cin >> path;
    m_gauss.loadMatrixFromFile(path);
    m_isCorrectLoad = true;
    std::cout << "\nDownload successful!\n";
    std::cout << "\nPress Enter to continue...\n";
    system("read REPLY");
    system("clear");
  } catch (std::exception &e) {
    printError(e.what());
  }
}

auto GaussInterface::viewMatrix() -> void {
  m_gauss.getMatrix().print();
  std::cout << "\nPress Enter to continue...\n";
  system("read REPLY");
  system("clear");
}

auto GaussInterface::viewGausAlrorithmWithoutParallel() -> void {
  printVector(m_gauss.getResultWithoutParallelAlgorithm());
}

auto GaussInterface::viewGausAlrorithmWithParallel() -> void {
  printVector(m_gauss.getResultWithParallelAlgorithm());
}

auto GaussInterface::printVector(const std::vector<double> &data) -> void {
  std::cout << std::endl;
  for (auto it = data.begin(); it != data.end(); ++it) {
    std::cout << *it << ", ";
  }

  std::cout << "\nPress Enter to continue...\n";
  system("read REPLY");
  system("clear");
}

auto GaussInterface::printError(const std::string &error) -> void {
  std::cout << "\nError! " << error << std::endl;
  std::cout << "\nPress Enter to continue...\n";
  system("read REPLY");
  system("clear");
}

auto GaussInterface::showWorkingTime() -> void {
  int n;
  std::cout << "\nEnter iteration quantity:" << std::endl << ">>> ";
  std::cin >> n;

  unsigned int start_time = clock();

  for (int i = 0; i < n; i++) {
    m_gauss.getResultWithoutParallelAlgorithm();
  }

  unsigned int end_time = clock();

  std::cout << "\nRunning time of the algorithm without parallel: "
            << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC
            << " sec";

  start_time = clock();

  for (int i = 0; i < n; i++) {
    m_gauss.getResultWithParallelAlgorithm();
  }

  end_time = clock();

  std::cout << "\nRunning time of the algorithm with parallel:    "
            << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC
            << " sec" << std::endl;

  std::cout << "\nPress Enter to continue...\n";
  system("read REPLY");
  viewMainMenu();
}

void GaussInterface::viewMainMenu() {
  try {
    system("clear");

    while (true) {
      std::cin.exceptions(std::ios_base::failbit);
      int action = mainMenuItem();

      switch (action) {
        case 1:
          loadMatrix();
          break;
        case 2:
          checkLoad();
          viewMatrix();
          break;
        case 3:
          checkLoad();
          viewGausAlrorithmWithoutParallel();
          break;
        case 4:
          checkLoad();
          viewGausAlrorithmWithParallel();
          break;
        case 5:
          checkLoad();
          showWorkingTime();
          break;
        case 0:
          system("clear");
          exit(EXIT_SUCCESS);
          break;

        default:
          std::cout << "Choose the correct action" << std::endl;
          std::cout << "\nPress Enter to continue...\n";
          system("read REPLY");
          viewMainMenu();
          break;
      }
    }

    system("clear");
  } catch (std::exception &e) {
    std::cout << "\n Check the correctness of the input" << std::endl;
    std::cin.clear();
    std::cin.ignore(10, '\n');
    std::cout << "\nPress Enter to continue...\n";
    system("read REPLY");
    viewMainMenu();
  }
}

auto GaussInterface::checkLoad() -> void {
  if (!m_isCorrectLoad) {
    std::cout << "\nMatrix not loaded\n";
    std::cout << "\nPress Enter to continue...\n";
    system("read REPLY");
    viewMainMenu();
  }
}

}  // namespace s21
