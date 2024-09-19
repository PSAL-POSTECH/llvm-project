//===- CFGLoopInfo.h - LoopInfo analysis for region bodies ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CFGLoopInfo analysis for MLIR. The CFGLoopInfo is used
// to identify natural loops and determine the loop depth of various nodes of a
// CFG.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_TILE_OPERATION_GRAPH_H
#define MLIR_ANALYSIS_TILE_OPERATION_GRAPH_H

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "mlir/IR/Operation.h"

namespace mlir {
class TOGLoopNode;
class TOGDMANode;
class TOGComputeNode;
} // namespace mlir

namespace mlir {
/// Representation of a single loop formed by blocks. The inherited LoopBase
/// class provides accessors to the loop analysis.
class TOGNode {
private:
  int node_id;
  std::string node_name;
 Operation* op; // Corresponding operation
  static int unique_id;
public:
  std::vector<TOGNode *> parents;
  std::vector<TOGNode *> children;
  // Constructor
  TOGNode(int id, const std::string &name) : node_id(id), node_name(name) {}
  TOGNode(const std::string &name) : node_id(createUniqueId()), node_name(name) {}
  int getId() const { return node_id; }
  static int createUniqueId() { return unique_id++; }
  std::string getName() const { return node_name; }
  std::vector<TOGNode*> getParents() const { return parents; }
  std::vector<TOGNode*> getChildren() const { return children; }
  TOGNode* getLastChild() const {
    if (children.size())
      return children.back();
    return NULL;
  }
  void setId(int id) { node_id = id; }
  void setName(const std::string &name) { node_name = name; }
  void addParent(TOGNode* parent) { parents.push_back(parent); }
  void addChild(TOGNode* child) { children.push_back(child); }
  Operation* getOp() { return op; }
  void setOp(Operation* op) { this->op = op; }
  enum NodeKind {
      BaseNodeKind,
      ComputeNodeKind,
      LoopNodeKind,
      DMANodeKind,
  };
  template<typename T>
  void printLoopInfo(std::string& name, const std::vector<T>& vec) const {
    std::cout << "\t\"" << name << "\": [";
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i];
      if (i < vec.size() - 1) {
        std::cout << ",";
      }
    }
    std::cout << "]";
  }
  void printLoopNodeInfo(std::string& name, const std::vector<TOGNode*>& vec) const {
    std::cout << "\t\"" << name << "\": [";
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i]->getId();
      if (i < vec.size() - 1) {
        std::cout << ",";
      }
    }
    std::cout << "]";
  }
  // Display node information
  virtual void display() const {
    std::cout << getId() <<" : {\n";
    std::cout << "\t\"node_id\": " << getId() << ",\n";
    std::cout << "\t\"node_name\": \"" << getName() << "\",\n";
    std::cout << "\t\"node_type\": " << getKind() << ",\n";
    auto name = std::string("parents");
    printLoopNodeInfo(name, this->parents);
    std::cout << ",\n";
    name = std::string("children");
    printLoopNodeInfo(name, this->children);
    if (getKind() == BaseNodeKind)
      std::cout << "\n}\n";
  }
  virtual NodeKind getKind() const { return BaseNodeKind; }
  static bool classof(const TOGNode *node) {
    // Check if the node kind matches this class's kind
    return node->getKind() == BaseNodeKind;
  }
  void bfs() {
    if (getName() == "root")
      std::cout << "graph = {\n";
    display();
    std::cout << ",";
    for (TOGNode *child: children) {
      child->bfs();
    }
    if (getName() == "root")
      std::cout << "}";

  }
};

class TOGLoopNode : public TOGNode {
private:
  std::string loop_idx;
  int loop_start;
  int loop_end;
  int loop_step;
  std::string loop_type;
 public:
  // Constructor
  TOGLoopNode(const std::string &name, const std::string &idx, int start, int end, int step, std::string &loop_type)
      : TOGNode(name), loop_idx(idx), loop_start(start), loop_end(end), loop_step(step), loop_type(loop_type) {}
  std::string getLoopIdx() const { return loop_idx; }
  int getLoopStart() const { return loop_start; }
  int getLoopEnd() const { return loop_end; }
  int getLoopStep() const { return loop_step; }
  std::string getLoopType() const { return loop_type; }

  // Setters
  void setLoopIdx(const std::string &idx) { loop_idx = idx; }
  void setLoopStart(int start) { loop_start = start; }
  void setLoopEnd(int end) { loop_end = end; }
  void setLoopStep(int step) { loop_step = step; }
  void setLoopType(std::string type) { loop_type = type; }
  // Display node information
  void display() const override {
    TOGNode::display();
    std::cout << ",\n";
    std::cout << "\t\"loop_index\": \"" << loop_idx << "\",\n";
    std::cout << "\t\"loop_start\": " << loop_start << ",\n";
    std::cout << "\t\"loop_end\": " << loop_end << ",\n";
    std::cout << "\t\"loop_step\": " << loop_step << ",\n";
    std::cout << "\t\"loop_type\": \"" << loop_type << "\"\n";
    std::cout << "}\n";
  }
  NodeKind getKind() const override { return LoopNodeKind; }
  static bool classof(const TOGNode *node) {
    // Check if the node kind matches this class's kind
    return node->getKind() == LoopNodeKind;
  }
};

class TOGDMANode : public TOGNode {
  std::string base_addr;
  std::vector<int> stride_list;
  std::vector<int> tile_size;
  std::vector<int> tile_stride;
  int element_size;
  bool is_write;

 public:
  TOGDMANode(const std::string &name, const std::string &addr,
             const std::vector<int> &strides, const std::vector<int> &sizes,
             const std::vector<int> &tileStrides, int elemSize, bool is_write)
      : TOGNode(name), base_addr(addr), stride_list(strides), tile_size(sizes),
        tile_stride(tileStrides), element_size(elemSize), is_write(is_write) {}
  std::string getBaseAddr() const { return base_addr; }
  std::vector<int> getStrideList() const { return stride_list; }
  std::vector<int> getTileSize() const { return tile_size; }
  std::vector<int> getTileStride() const { return tile_stride; }
  int getElementSize() const { return element_size; }
  bool isWrite() const { return is_write; }
  void setBaseAddr(const std::string &addr) { base_addr = addr; }
  void setStrideList(const std::vector<int> &strides) { stride_list = strides; }
  void setTileSize(const std::vector<int> &sizes) { tile_size = sizes; }
  void setTileStride(const std::vector<int> &tileStrides) { tile_stride = tileStrides; }
  void setElementSize(int elemSize) { element_size = elemSize; }
  void setIsWrite(bool is_write) const { is_write = is_write; }
  void display() const override {
    TOGNode::display();
    std::cout << ",\n";
    std::cout << "\t\"is_write\": " << is_write << ",\n";
    std::cout << "\t\"base_address\": \"" << base_addr << "\",\n";

    auto name = std::string("stride_list");
    printLoopInfo(name, this->stride_list);
    std::cout << ",\n";
    name = std::string("tile_stride");
    printLoopInfo(name, this->tile_stride);
    std::cout << ",\n";

    name = std::string("tile_size");
    printLoopInfo(name, this->tile_size);
    std::cout << ",\n";
    std::cout << "\t\"element_size\": " << element_size << "\n";
    std::cout << "}\n";
  }
  NodeKind getKind() const override { return DMANodeKind; }
  static bool classof(const TOGNode *node) {
    // Check if the node kind matches this class's kind
    return node->getKind() == DMANodeKind;
  }
};

class TOGComputeNode : public TOGNode {
public:
  enum ComputeType{
    VectorCompute,
    MatmulCompute
  };
  std::vector<Operation*> operations;
  TOGComputeNode(const std::string &name, int cycle, ComputeType type) :
    TOGNode(name), compute_cycle(cycle), compute_type(type) {}
  int getComputeCycle() const { return compute_cycle; }
  ComputeType getComputeType() const { return compute_type; }
  void setComputeCycle(int cycle) { compute_cycle = cycle; }
  void setComputeType(ComputeType type) { compute_type = type; }
  void display() const override {
    TOGNode::display();
    std::cout << ",\n";
    std::cout << "\t\"compute_cycle\": " << compute_cycle << ",\n";
    std::cout << "\t\"compute_type\": " << compute_type << "\n";
    std::cout << "}\n";
  }
  NodeKind getKind() const override { return ComputeNodeKind; }
  static bool classof(const TOGNode *node) {
    // Check if the node kind matches this class's kind
    return node->getKind() == ComputeNodeKind;
  }
private:
  int compute_cycle;
  ComputeType compute_type;
};

int TOGNode::unique_id = 0;

} // namespace mlir

#endif // MLIR_ANALYSIS_TILE_OPERATION_GRAPH_H
