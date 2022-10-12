/**
 * @file    ants_parallel_funcs.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "ants_colony.h"

namespace s21 {

auto AntsColony::FindPathsParalell() -> void {
  auto function = [](Matrix<double> &pheromons, Graph &graph,
                     std::vector<Ant> &ants, int remainder) {
    for (int i = 0; i < ants.size(); i++) {
      if (i % graph.size() == remainder) ants[i].FindPath(graph, pheromons);
    }
  };
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i] = std::thread(function, std::ref(pheromons_), std::ref(graph_),
                              std::ref(ants_), i);
    threads_[i].join();
  }
}
}  // namespace s21