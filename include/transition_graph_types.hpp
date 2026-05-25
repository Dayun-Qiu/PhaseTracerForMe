// ====================================================================
// This file is part of PhaseTracer

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// ====================================================================

#ifndef PHASETRACER_TRANSITION_GRAPH_TYPES_HPP_
#define PHASETRACER_TRANSITION_GRAPH_TYPES_HPP_

#include <limits>
#include <string>
#include <vector>

namespace TransitionGraph {

struct TransitionEdge {
  int transitionIndex;
  bool subcritical;
  double temperature;

  TransitionEdge(int transitionIndex, bool subcritical, double temperature) {
    this->transitionIndex = transitionIndex;
    this->subcritical = subcritical;
    this->temperature = temperature;
  }

  friend std::ostream &operator<<(std::ostream &o, const TransitionEdge &te) {
    o << "{i: " << te.transitionIndex << ", sc: " << te.subcritical << ", T: " << te.temperature << "}";
    return o;
  }
};

struct Edge {
  int fromPhase;
  int toPhase;
  double temperature;
  int transition;

  Edge(int fromPhase, int toPhase, double temperature, int transition) {
    this->fromPhase = fromPhase;
    this->toPhase = toPhase;
    this->temperature = temperature;
    this->transition = transition;
  }

  friend std::ostream &operator<<(std::ostream &o, const Edge &e) {
    o << e.fromPhase << " --(" << e.transition << ", T=" << e.temperature << ")--> " << e.toPhase;
    return o;
  }
};

struct Vertex {
  int phase;
  std::vector<Edge> edges;

  Vertex(int phase) {
    this->phase = phase;
    this->edges = {};
  }

  void addEdge(Edge &edge) {
    assert(edge.fromPhase == phase && "Attempted to add an edge to a vertex that is not the start point of the edge!");
    edges.push_back(edge);
  }

  friend std::ostream &operator<<(std::ostream &o, const Vertex &v) {
    o << "Vertex<" << v.phase << "> {";

    if (v.edges.size() == 0) {
      o << "}";
      return o;
    }

    for (int i = 0; i < v.edges.size() - 1; ++i) {
      o << v.edges[i] << ", ";
    }

    if (v.edges.size() > 0) {
      o << v.edges.back();
    }

    o << "}";
    return o;
  }
};

struct Path {
  std::vector<int> phases;
  std::vector<TransitionEdge> transitions;

  Path(int startPhase) {
    phases.push_back(startPhase);
  }

  void extend(const Edge &edge, bool subcritical, double temperature) {
    assert(edge.fromPhase == phases.back() && "Attempted to extend the path using an edge that doesn't begin at the end of this path!");

    phases.push_back(edge.toPhase);
    transitions.push_back({edge.transition, subcritical, temperature});
  }

  bool canUndergoTransition(const std::vector<Vertex> &vertices, int vertexIndex, int edgeIndex) {
    assert(vertexIndex == phases.back() && "Checked whether a transition was possible from a phase that isn't the end of this path!");

    const Vertex &vertex = vertices[vertexIndex];
    const Edge &transition = vertex.edges[edgeIndex];

    int fromPhase = vertex.edges[edgeIndex].fromPhase;
    int toPhase = vertex.edges[edgeIndex].toPhase;
    double currentTemperature = transitions.back().temperature;

    if (transition.temperature <= currentTemperature) {
      return true;
    }

    const Vertex &toVertex = vertices[transition.toPhase];

    for (int i = 0; i < toVertex.edges.size(); ++i) {
      if (toVertex.edges[i].temperature < currentTemperature) {
        continue;
      }

      if (toVertex.edges[i].temperature > transition.temperature) {
        break;
      }

      if (toVertex.edges[i].toPhase == vertex.phase) {
        return false;
      }
    }

    for (int i = 0; i < phases.size(); ++i) {
      if (phases[i] == toPhase) {
        return false;
      }
    }

    return true;
  }

  double getCurrentTemperature() {
    return transitions.size() > 0 ? transitions.back().temperature : std::numeric_limits<double>::max();
  }

  friend std::ostream &operator<<(std::ostream &o, const Path &p) {
    if (p.phases.size() == 0) {
      o << "<empty path>";
      return o;
    }

    o << p.phases[0];

    for (int i = 1; i < p.phases.size(); ++i) {
      o << " --" << p.transitions[i - 1] << "--> " << p.phases[i];
    }

    return o;
  }

  std::string getStringForFileOutput() {
    std::string transitionIndices = "";

    if (transitions.size() == 0) {
      return transitionIndices;
    }

    transitionIndices += std::to_string(transitions[0].transitionIndex);

    for (int i = 1; i < transitions.size(); ++i) {
      transitionIndices += " " + std::to_string(transitions[i].transitionIndex);
    }

    return transitionIndices;
  }
};

struct FrontierNode {
  int vertexIndex;
  int edgeIndex;
  int pathIndex;

  FrontierNode(int vertexIndex, int edgeIndex, int pathIndex) {
    this->vertexIndex = vertexIndex;
    this->edgeIndex = edgeIndex;
    this->pathIndex = pathIndex;
  }

  friend std::ostream &operator<<(std::ostream &o, const FrontierNode &fn) {
    o << "{v: " << fn.vertexIndex << ", e: " << fn.edgeIndex << ", p: " << fn.pathIndex << "}";
    return o;
  }
};

} // namespace TransitionGraph

#endif // PHASETRACER_TRANSITION_GRAPH_TYPES_HPP_
