/**
 * @file    ant.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "ant.h"

#include <algorithm>

namespace s21 {

Ant::Ant(size_t vertex) : vertex_(vertex) {
  visitedVertices_.push_back(vertex_);
}

auto Ant::GetPath() -> std::vector<int> & { return visitedVertices_; }

auto Ant::CanMove(Graph &graph) -> bool {
  possibleWays_.clear();
  for (int i = 0; i < graph.size(); i++) {
    if (i != vertex_ && graph(vertex_, i) != 0 &&
        !ContainsIn(i, visitedVertices_)) {
      possibleWays_.emplace_back(i);
    }
  }
  return !possibleWays_.empty();
}

auto comp(const std::pair<int, double> &p1, const std::pair<int, double> &p2)
    -> bool {
  return p1.second < p2.second;
}

auto Ant::FindPath(Graph &graph, const Matrix<double> &pheromons) -> void {
  while (CanMove(graph)) {
    double sum{0};
    std::vector<std::pair<int, double>> probabilities;
    for (auto &to : possibleWays_) {
      double pathAttraction = pow(graph(vertex_, to), alpha_);
      double amountOfPheromone = pow(pheromons(vertex_, to), beta_);
      double p = pathAttraction * amountOfPheromone;
      sum += p;
      probabilities.emplace_back(to, p);
    }
    for (auto &elem : probabilities) {
      elem.second = elem.second / sum;
    }
    double random = RandomIntInDist(100) / 100;
    std::sort(probabilities.begin(), probabilities.end(), comp);

    auto lesser = [random](const std::pair<int, double> &elem) {
      return random < elem.second;
    };
    auto elem =
        std::find_if(probabilities.begin(), probabilities.end(), lesser);

    if (elem != probabilities.end()) {
      visitedVertices_.emplace_back((*elem).first);
      vertex_ = (*elem).first;
    }
  }
  if (graph(visitedVertices_.back(), visitedVertices_[0]) != 0 &&
      static_cast<int>(visitedVertices_.size()) == graph.size()) {
    visitedVertices_.emplace_back(visitedVertices_[0]);
  } else {
    visitedVertices_.clear();
  }
}

}  // namespace s21
