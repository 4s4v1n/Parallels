/**
 * @file    gauss_interface.h
 * @authors telvina<at>student.21-school.ru
 * @authors sreanna<at>student.21-school.ru
 * @authors jmadie<at>student.21-school.ru
 * @date    13/09/22
 */

#pragma once

#include <ctime>

#include "gauss.h"

namespace s21 {

/**
 * @class Gauss Algorithm Interface
 * @brief Implements simple interface in cmd
 */

class GaussInterface {
 private:
  Gauss m_gauss;
  bool m_isCorrectLoad = false;

  auto mainMenuItem() -> int;

  auto loadMatrix() -> void;

  auto viewMatrix() -> void;

  auto viewGausAlrorithmWithoutParallel() -> void;

  auto viewGausAlrorithmWithParallel() -> void;

  auto showWorkingTime() -> void;

  auto printVector(const std::vector<double> &data) -> void;

  auto printError(const std::string &error) -> void;

  auto checkLoad() -> void;

 public:
  GaussInterface() {}

  ~GaussInterface() {}

  auto viewMainMenu() -> void;
};

}  // namespace s21