/**
 * @file    ant.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef S21_ANT_H
#define S21_ANT_H

#include <vector>

#include "../graph/s21_graph.h"
#include "common_funcs.h"

/**
 * @class   Winograd
 * @brief   Defines class for ant like in real world
 */

namespace s21 {

class Ant {
 public:
  explicit Ant(size_t vertex);

  Ant() = delete;

  auto FindPath(Graph &graph, const Matrix<double> &feromons) -> void;

  [[nodiscard]] auto GetPath() -> std::vector<int> &;

  auto CanMove(Graph &graph) -> bool;

 private:
  int vertex_{-1};
  /// alpha_ - множитель привлекательости пути, beta_ - множитель феромона
  double alpha_{2}, beta_{1};
  std::vector<int> visitedVertices_;
  std::vector<int> possibleWays_;
};

}  // namespace s21

#endif  // S21_ANT_H
