/**
 * @file    common_funcs.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef COMMON_FUNCS_H
#define COMMON_FUNCS_H
#include <random>
#include <vector>

namespace s21 {
auto ContainsIn(int vertex, const std::vector<int> &vertices) -> bool;

auto RandomIntInDist(size_t dist) -> int;
}  // namespace s21
#endif  // COMMON_FUNCS_H
