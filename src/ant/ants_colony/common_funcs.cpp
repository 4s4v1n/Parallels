/**
 * @file    common_funcs.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "common_funcs.h"

#include <algorithm>

namespace s21 {

auto ContainsIn(int vertex, const std::vector<int> &vertices) -> bool {
  auto equality = [vertex](int elem) { return elem == vertex; };
  return std::any_of(vertices.begin(), vertices.end(), equality);
}

auto RandomIntInDist(size_t dist) -> int {
  std::random_device rd;
  std::default_random_engine engine{rd()};
  std::uniform_int_distribution<int> distribution{0,
                                                  static_cast<int>(dist) - 1};
  return distribution(engine);
}

}  // namespace s21