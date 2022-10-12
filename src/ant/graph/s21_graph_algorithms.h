/**
 * @file    s21_graph_algorithms.h
 * @authors telvina<at>student.21-school.ru
 * @authors sreanna<at>student.21-school.ru
 * @authors jmadie<at>student.21-school.ru
 * @date    04/09/22
 */

#pragma once

#include <vector>

#include "s21_graph.h"

/**
 * @class GraphAlgorithms
 * @brief This class holds the methods of algorithms on graphs
 */

struct TspResult {
  std::vector<int> vertices;
  double distance;
  TspResult() : distance(0) {}
};

namespace s21 {
class GraphAlgorithms {
 public:
  static auto solveTravelingSalesmanProblem(Graph &graph) -> TspResult;
  static auto solveTravelingSalesmanProblemParalell(Graph &graph) -> TspResult;
};
}  // namespace s21
