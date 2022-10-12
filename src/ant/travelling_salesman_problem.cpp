/**
 * @file    travelling_salesman_problem.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "./ants_colony/ants_colony.h"
#include "./graph/s21_graph_algorithms.h"

namespace s21 {

auto GraphAlgorithms::solveTravelingSalesmanProblem(Graph &graph) -> TspResult {
    int antsNumber = 12;
    int CountWithoutUpgrade = 30, maxCount = 50;

    s21::AntsColony colony(graph);
    TspResult prevRes, res;
    prevRes.distance = INFINITY;
    res.distance = INFINITY;

    while (CountWithoutUpgrade) {
        colony.GenerateAnts(antsNumber);
        colony.FindPaths();
        colony.UpdateFeromon();
        colony.UpdateResult(res);

        if (res.distance == prevRes.distance) {
            CountWithoutUpgrade--;
        } else {
            prevRes = res;
            CountWithoutUpgrade = maxCount;
        }
    }
    return res;
}

auto GraphAlgorithms::solveTravelingSalesmanProblemParalell(Graph &graph) -> TspResult {
    int antsNumber = 12;
    int CountWithoutUpgrade = 30, maxCount = 50;

    s21::AntsColony colony(graph);
    TspResult prevRes, res;
    prevRes.distance = INFINITY;
    res.distance = INFINITY;

    while (CountWithoutUpgrade) {
        colony.GenerateAnts(antsNumber);
        colony.FindPathsParalell();
        colony.UpdateFeromon();
        colony.UpdateResult(res);

        if (res.distance == prevRes.distance) {
            CountWithoutUpgrade--;
        } else {
            prevRes = res;
            CountWithoutUpgrade = maxCount;
        }
    }
    return res;
}
}  // namespace s21