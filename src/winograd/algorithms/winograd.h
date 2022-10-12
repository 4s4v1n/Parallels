/**
 * @file    winograd.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "../../matrix/matrix.h"

#ifndef WINOGRAD_H
#define WINOGRAD_H

#include <queue>
#include <thread>
#include <vector>

#include "../types/run_type.h"

namespace s21 {

/**
 * @class   Winograd
 * @brief   Defines class with the solution of matrix multiplication with
 * different thread model
 */

class Winograd {
  using Factor = std::vector<double>;

 public:
  Winograd() = default;
  ~Winograd() = default;

 public:
  auto run(RunType type) -> const Matrix<double>&;
  auto setThreadsNumber(std::uint32_t number) -> void;
  auto setMatrices(const Matrix<double>& first, const Matrix<double>& second)
      -> void;

 private:
  auto calculateRowFactor(int i) -> void;
  auto calculateColFactor(int i) -> void;
  auto calculateRowValues(int i) -> void;
  auto calculateResultMatrix() -> void;

 private:
  auto runOneThread() -> void;
  auto runMultiplyThreads(
      uint32_t threadNumber = std::thread::hardware_concurrency()) -> void;
  auto runtMultiplyThreadsConveyor() -> void;

 private:
  auto preprocessing() -> void;
  auto oddMatrixProcessing() -> void;

 private:
  std::vector<std::thread> m_threads{};
  Matrix<double> m_first{};
  Matrix<double> m_second{};
  Matrix<double> m_result{};
  int m_halfSize{};
  Factor m_rowFactor{};
  Factor m_colFactor{};
  uint32_t m_threadNumber{};
};

};  // namespace s21

#endif  // WINOGRAD_H