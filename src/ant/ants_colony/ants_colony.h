/**
 * @file    ants_colony.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef S21_ANTS_COLONY_H
#define S21_ANTS_COLONY_H

#include <pthread.h>

#include <cmath>
#include <mutex>
#include <thread>

#include "../graph/s21_graph_algorithms.h"
#include "ant.h"
#include "common_funcs.h"

namespace s21 {

/**
 * @class   AntsColony
 * @brief   Defines class that simulates behavior of ants
 */

class AntsColony {
 public:
  AntsColony() = delete;

  explicit AntsColony(Graph &graph);

 public:
  auto GenerateAnts(int allAnts) -> void;

  auto FindPaths() -> void;

  auto UpdateFeromon() -> void;

  auto UpdateResult(TspResult &res) -> void;

  auto FindPathsParalell() -> void;

 private:
  auto CalculatePathCost(std::vector<int> &path) -> int;

  double pheromonRemains_ = 0.75;

  Graph &graph_;
  Matrix<double> pheromons_;
  std::vector<Ant> ants_;
  int allVertex_{-1};
  int averageRouteLen_;

  std::vector<std::thread> threads_;
};

}  // namespace s21

#endif  // S21_ANTS_COLONY_H
