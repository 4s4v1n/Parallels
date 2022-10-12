/**
 * @file    interface.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef S21_INTERFACE_H
#define S21_INTERFACE_H

#include <cstdlib>
#include <exception>
#include <iostream>

#include "../graph/s21_graph_algorithms.h"

namespace s21 {

enum { NORMAL, ERROR };

/**
 * @class   AntInterface
 * @brief   Defines class of console interface for ant algorithm
 */

class AntInterface {
 public:
  AntInterface();

  auto Show() -> void;

  auto LoadGraph() -> void;

  auto mainMenuItem() -> int;

  auto Research() -> void;

  auto SetNumberOfIterations() -> void;

 private:
  Graph m_graph = Graph(10);
  bool m_isCorrectGraphLoad;
  int m_number_of_iterations{10};
};
}  // namespace s21

#endif  // S21_INTERFACE_H