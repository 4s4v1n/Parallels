/**
 * @file    ant_colony.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "ants_colony.h"

namespace s21 {

AntsColony::AntsColony(Graph &graph) : graph_(graph) {
  allVertex_ = graph.size();
  pheromons_ = Matrix<double>(allVertex_, allVertex_);
  pheromons_.fill(0.2);
  averageRouteLen_ = 0;
  for (int i = 0; i < allVertex_; i++) {
    for (int j = 0; j < allVertex_; j++) {
      averageRouteLen_ += graph(i, j);
    }
  }
  averageRouteLen_ /= allVertex_ * allVertex_ - allVertex_;
  threads_.resize(4);
}

auto AntsColony::GenerateAnts(int allAnts) -> void {
  ants_.clear();
  for (int i = 0; i < allAnts; i++) {
    ants_.push_back(Ant(RandomIntInDist(allVertex_)));
  }
}

auto AntsColony::FindPaths() -> void {
  for (auto &ant : ants_) {
    ant.FindPath(graph_, pheromons_);
  }
}

auto AntsColony::CalculatePathCost(std::vector<int> &path) -> int {
  int res{};
  for (int i = 1; i < static_cast<int>(path.size()); i++) {
    res += graph_(path[i - 1], path[i]);
  }
  return res;
}

auto AntsColony::UpdateFeromon() -> void {
  for (Ant &ant : ants_) {
    std::vector<int> &path = ant.GetPath();
    int cost = CalculatePathCost(path);
    double deltaPheromone = static_cast<double>(averageRouteLen_) / cost;
    for (int i = 1; i < static_cast<int>(path.size()); i++) {
      pheromons_(path[i - 1], path[i]) *= pheromonRemains_;
      pheromons_(path[i - 1], path[i]) += deltaPheromone;
    }
  }
  for (int i = 0; i < pheromons_.rows(); i++) {
    for (int j = 0; j < pheromons_.cols(); j++) {
      pheromons_(i, j) *= pheromonRemains_;
    }
  }
}

auto AntsColony::UpdateResult(TspResult &res) -> void {
  for (auto &ant : ants_) {
    std::vector<int> &path = ant.GetPath();
    int cost = CalculatePathCost(path);
    if (cost < res.distance && cost) {
      res.distance = cost;
      res.vertices = path;
    }
  }
}

}  // namespace s21