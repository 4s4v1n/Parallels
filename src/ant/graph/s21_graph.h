/**
 * @file    s21_graph.h
 * @authors telvina<at>student.21-school.ru
 * @authors sreanna<at>student.21-school.ru
 * @authors jmadie<at>student.21-school.ru
 * @date    04/09/22
 */

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../matrix/matrix.h"

/**
 * @class  Graph
 * @brief  Implements mathematical model of graph by adjacency matrix
 */

class Graph {
 public:
  explicit Graph(int size);
  Graph(const Graph& other);
  Graph(Graph&& other) noexcept;
  ~Graph() = default;

 public:
  auto loadGraphFromFile(std::string filename) -> void;
  auto exportGraphToDot(const std::string& filename) -> void;
  auto printGraph() -> void;
  auto IsComplete() -> bool;

 public:
  [[nodiscard]] auto size() const -> int;
  [[nodiscard]] auto adjacencyMatrix() const -> const Matrix<int>&;
  [[nodiscard]] auto isWeighted() const -> bool;
  [[nodiscard]] auto isDirected() const -> bool;

 public:
  auto operator=(const Graph& other) -> Graph&;
  auto operator()(int row, int column) -> int&;
  auto operator=(Graph&& other) -> Graph&;

 private:
  auto setSize(const int size) -> void;
  auto typeCheck() -> void;
  auto typeCheckForWeightedGraph() -> void;
  auto exportUndirectedGraphToDot(const std::string& filename) -> void;
  auto exportDirectedGraphToDot(const std::string& filename) -> void;

 private:
  Matrix<int> m_matrix{};
  int m_size{};
  bool m_isWeightedGraph{};
  bool m_isDirectedGraph{};

 private:
  static constexpr int charBias = 97;
};
