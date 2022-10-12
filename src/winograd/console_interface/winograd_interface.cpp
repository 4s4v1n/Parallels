/**
 * @file    winograd_interface.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "winograd_interface.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <thread>

namespace s21 {

auto WinogradInterface::mainMenu() -> int {
  std::cin.exceptions(std::ios_base::failbit);
  int action{};
  std::cout << "___________________________________________________________\n"
            << "|                                                         |\n"
            << "|                     Choose action:                      |\n"
            << "|                                                         |\n"
            << "| 1. Set matrices                                         |\n"
            << "| 2. Set repeat number                                    |\n"
            << "| 3. Run time tests                                       |\n"
            << "|                                                         |\n"
            << "| 0. Exit                                                 |\n"
            << "|_________________________________________________________|\n"
            << std::endl
            << ">>> ";
  std::cin >> action;

  return action;
}

auto WinogradInterface::show() -> void {
  try {
    system("clear");

    while (true) {
      std::cin.exceptions(std::ios_base::failbit);
      int action = mainMenu();
      switch (action) {
        case 1: {
          setMatrices();
          break;
        }
        case 2: {
          setRepeatNumber();
          break;
        }
        case 3: {
          runAllCases();
          break;
        }
        case 0: {
          system("clear");
          exit(EXIT_SUCCESS);
        }
        default: {
          std::cout << "Choose the correct action" << std::endl;
          system("read REPLY");
          show();
          break;
        }
      }
      system("clear");
    }
  } catch (const std::exception& exception) {
    printError(exception);
    std::cout << "\n Check the correctness of the input" << std::endl;
    std::cin.clear();
    std::cin.ignore(10, '\n');
    system("read REPLY");
    show();
  }
}

auto WinogradInterface::printError(const std::exception& error) -> void {
  std::cout << "\nError! " << error.what() << std::endl;
  system("read REPLY");
  system("clear");
}

auto WinogradInterface::setMatrices() -> void {
  Matrix<double> first{};
  Matrix<double> second{};
  int rows{};
  int cols{};
  std::cout << "Set first matrix rows: ";
  std::cin >> rows;
  std::cout << "Set first matrix cols: ";
  std::cin >> cols;
  first.resize(rows, cols);
  std::cout << "Set second matrix rows: ";
  std::cin >> rows;
  std::cout << "Set second matrix cols: ";
  std::cin >> cols;
  second.resize(rows, cols);
  system("clear");
  char action{};
  std::cout << "Fill matrices automatically?(y/n): ";
  std::cin >> action;
  if (action == 'y') {
    first.rand(1, 9);
    second.rand(1, 9);
  } else {
    std::cout << "Fill first matrix:" << std::endl;
    for (int i{0}; i < first.rows(); ++i) {
      for (int j{0}; j < first.cols(); ++j) {
        std::cin >> first(i, j);
      }
    }
    std::cout << "Fill second matrix:" << std::endl;
    for (int i{0}; i < second.rows(); ++i) {
      for (int j{0}; j < second.cols(); ++j) {
        std::cin >> second(i, j);
      }
    }
    system("clear");
  }
  m_winograd.setMatrices(first, second);
  m_bMatricesSet = true;
  system("clear");
}

auto WinogradInterface::setRepeatNumber() -> void {
  std::cout << "Input repeat number: ";
  std::cin >> m_nRepeat;
  system("clear");
}

auto WinogradInterface::runCase(const std::string& message, RunType type)
    -> void {
  std::cout << "___________________________________________________________\n";
  std::cout << message << std::endl;
  Matrix<double> result{};
  if (m_nRepeat == 0) m_nRepeat = 1;
  auto begin{std::chrono::high_resolution_clock::now()};
  for (int i{0}; i < m_nRepeat; ++i) {
    result = m_winograd.run(type);
  }
  auto end{std::chrono::high_resolution_clock::now()};
  std::cout << "Time: " << std::chrono::duration<double>(end - begin).count()
            << "sec" << std::endl;
  std::cout << "___________________________________________________________\n";
  showResult(result);
}

auto WinogradInterface::runAllCases() -> void {
  if (m_bMatricesSet) {
    try {
      runCase("Single thread", RunType::SINGLE);
      runCase("Multithreading classic", RunType::MULTITHREADING_CLASSIC);
      runCase("Multithreading conveyor", RunType::MULTITHREADING_CONVEYOR);
      for (std::uint32_t i{2}; i <= 4 * std::thread::hardware_concurrency();
           i *= 2) {
        m_winograd.setThreadsNumber(i);
        runCase("Multithreading class with " + std::to_string(i) + " threads",
                RunType::MULTITHREADING_NUMBER);
      }
    } catch (const std::exception& exception) {
      std::cout << exception.what() << std::endl;
    }
  } else {
    std::cout << "Matrices are not specified" << std::endl;
  }
}

auto WinogradInterface::showResult(Matrix<double>& result) -> void {
  char action{};
  std::cout << "Show result?(y/n): ";
  std::cin >> action;
  if (action != 'y') return;
  for (int i{0}; i < result.rows(); ++i) {
    for (int j{0}; j < result.cols(); ++j) {
      std::cout << result(i, j) << ' ';
    }
    std::cout << std::endl;
  }
}

}  // namespace s21
