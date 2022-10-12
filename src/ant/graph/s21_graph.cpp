/**
 * @file    s21_graph.cpp
 * @authors telvina<at>student.21-school.ru
 * @authors sreanna<at>student.21-school.ru
 * @authors jmadie<at>student.21-school.ru
 * @date    04/09/22
 */

#include "s21_graph.h"

Graph::Graph(int size)
    : m_size{size},
      m_matrix{size, size},
      m_isDirectedGraph{false},
      m_isWeightedGraph{false} {}

Graph::Graph(const Graph& other) { *this = other; }

Graph::Graph(Graph&& other) noexcept { *this = std::move(other); }

Graph& Graph::operator=(const Graph& other) {
  if (this != &other) {
    m_size = other.m_size;
    m_matrix = other.m_matrix;
    m_isWeightedGraph = other.m_isWeightedGraph;
    m_isDirectedGraph = other.m_isDirectedGraph;
  }
  return *this;
}

auto Graph::operator=(Graph&& other) -> Graph& {
  if (this != &other) {
    std::swap(m_size, other.m_size);
    std::swap(m_matrix, other.m_matrix);
    std::swap(m_isWeightedGraph, other.m_isWeightedGraph);
    std::swap(m_isDirectedGraph, other.m_isDirectedGraph);
  }
  return *this;
}

auto Graph::printGraph() -> void {
  std::cout << "Size: " << m_size << std::endl;
  if (!m_isDirectedGraph) {
    std::cout << "Type: undirected" << std::endl;
  } else {
    std::cout << "Type: directed" << std::endl;
  }
  std::cout << "Weighted graph: " << m_isWeightedGraph << std::endl;
  std::cout << std::endl;
  for (int i{0}; i < m_size; ++i) {
    for (int j{0}; j < m_size; ++j) {
      std::cout << m_matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

auto Graph::setSize(const int size) -> void {
  if (size != m_size) {
    Graph tempGraph(size);
    *this = tempGraph;
  }
}

auto Graph::loadGraphFromFile(std::string filename) -> void {
  std::ifstream fs;
  fs.open(filename, std::fstream::in);
  if (!fs.is_open()) {
    throw std::runtime_error{"Incorrect file path"};
  }
  int fileSize;
  fs >> fileSize;
  setSize(fileSize);
  for (int i{0}; i < m_size; ++i) {
    for (int j{0}; j < m_size; ++j) {
      if (fs.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error{"Incorrect file"};
      }

      fs >> m_matrix(i, j);

      if (!m_isWeightedGraph && m_matrix(i, j) > 1) {
        m_isWeightedGraph = true;
      }

      if (!m_isDirectedGraph &&
          m_matrix(i, j) != m_matrix(j, i)) {  //  && i != j
        m_isDirectedGraph = true;
      }
    }
  }

  fs.close();
}

auto Graph::exportGraphToDot(const std::string& filename) -> void {
  std::string filePath{filename + ".dot"};
  std::fstream outfile;
  outfile.open(filePath, std::fstream::out);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to write file");
  }
  std::string dotSeparator;
  std::string dotGraphType;
  if (m_isDirectedGraph) {
    dotSeparator = " -> ";
    dotGraphType = "digraph ";
  } else {
    dotSeparator = " -- ";
    dotGraphType = "graph ";
  }
  outfile << dotGraphType << "dotfile"
          << " {" << std::endl;
  for (int i{0}; i < m_size; ++i) {
    for (int j{0}; j < m_size; ++j) {
      if (m_matrix(i, j) > 0) {
        if (m_matrix(i, j) == 1) {
          outfile << i + 1 << dotSeparator << j + 1 << ";" << std::endl;
        } else {
          outfile << i + 1 << dotSeparator << j + 1
                  << " [label=" << m_matrix(i, j) << "];" << std::endl;
        }
      }
    }
  }

  outfile << "}";
}

auto Graph::exportUndirectedGraphToDot(const std::string& filename) -> void {
  std::string filePath{filename + ".dot"};
  std::fstream outfile;
  outfile.open(filePath, std::fstream::out);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to open file");
  }
  outfile << "graph dotfile"
          << " {" << std::endl;
  for (int i{0}; i < m_size; ++i) {
    for (int j{0}; j < m_size; ++j) {
      if (m_matrix(i, j) > 0) {
        char first = i + 97;
        char second = j + 97;
        if (m_matrix(i, j) == 1) {
          outfile << first << " -- " << second << ";" << std::endl;
        } else {
          outfile << first << " -- " << second << " [label=" << m_matrix(i, j)
                  << "];" << std::endl;
        }
      }
    }
  }

  outfile << "}";
}

auto Graph::exportDirectedGraphToDot(const std::string& filename) -> void {
  std::string filePath = {filename + ".dot"};
  std::fstream outfile;
  outfile.open(filePath, std::fstream::out);
  if (!outfile.is_open()) {
    throw std::runtime_error("Unable to open file");
  }
  outfile << "graph " << filename << " {" << std::endl;
  for (int i = 0; i < m_size; i++) {
    for (int j = 0; j < m_size; j++) {
      if (m_matrix(i, j) > 0) {
        char first = static_cast<char>(i + charBias);
        char second = static_cast<char>(j + charBias);
        if (m_matrix(i, j) == 1) {
          outfile << first << " -> " << second << ";" << std::endl;
        } else {
          outfile << first << " -> " << second << " [label=" << m_matrix(i, j)
                  << "];" << std::endl;
        }
      }
    }
  }

  outfile << "}";
}

auto Graph::operator()(int row, int column) -> int& {
  return m_matrix.operator()(row, column);
}

auto Graph::adjacencyMatrix() const -> const Matrix<int>& { return m_matrix; }

auto Graph::size() const -> int { return m_size; }

auto Graph::isWeighted() const -> bool { return m_isWeightedGraph; }

auto Graph::isDirected() const -> bool { return m_isDirectedGraph; }

auto Graph::IsComplete() -> bool {
  for (int i = 0; i < m_size; i++) {
    for (int j = 0; j < m_size; j++) {
      if (m_matrix(i, j) == 0 && i != j) return false;
    }
  }
  return true;
}