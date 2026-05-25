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

#ifndef PHASETRACER_TRANSITION_GRAPH_UTIL_HPP_
#define PHASETRACER_TRANSITION_GRAPH_UTIL_HPP_

#include <assert.h>
#include <stdlib.h>

#include <vector>

#include "phase_finder.hpp"
#include "transition_graph_types.hpp"

// Forward declarations.
namespace PhaseTracer {
struct Transition;
class TransitionFinder;
} // namespace PhaseTracer

namespace TransitionGraph {

/**
 * Stores the boundary conditions of the phase history (i.e. the phases that exist at the high temperature, and the
 * lowest energy phases at the low temperature (T=0)), and whether the model is valid at the low temperature (i.e.
 * whether the lowest energy phase is in the expected field configuration).
 *
 * This struct is constructed in -- and returned from -- TransitionGraph::extractPhaseStructureData.
 */
struct PhaseStructureData {
  std::vector<int> lowTPhaseIndices;
  std::vector<int> highTPhaseIndices;
  std::vector<bool> isLowTemperaturePhase;
  bool validAtZeroT;

  PhaseStructureData(std::vector<int> lowTPhaseIndices, std::vector<int> highTPhaseIndices, bool validAtZeroT,
                     std::vector<bool> isLowTemperaturePhase) {
    this->lowTPhaseIndices = lowTPhaseIndices;
    this->highTPhaseIndices = highTPhaseIndices;
    this->validAtZeroT = validAtZeroT;
    this->isLowTemperaturePhase = isLowTemperaturePhase;
  }

  friend std::ostream &operator<<(std::ostream &o, const PhaseStructureData &p) {
    o << "Valid at T=0: " << p.validAtZeroT << ", low-T phases: {";

    if (p.lowTPhaseIndices.size() > 0) {
      o << p.lowTPhaseIndices[0];

      for (int i = 1; i < p.lowTPhaseIndices.size(); ++i) {
        o << " " << p.lowTPhaseIndices[i];
      }
    }

    o << "}, high-T phases: {";

    if (p.highTPhaseIndices.size() > 0) {
      o << p.highTPhaseIndices[0];

      for (int i = 1; i < p.highTPhaseIndices.size(); ++i) {
        o << " " << p.highTPhaseIndices[i];
      }
    }

    o << "}";
    return o;
  }
};

/**
 * Constructs a new phase from the input phase, with the position vector X being reflected about the axes defined in
 * reflectionIndices. For instance, if reflectionIndices = {0, 2} and phase.X[i] = (x0, x1, x2, x3), then
 * newPhase.X[i] = (-x0, x1, -x2, x3).
 *
 * @param phase - The phase explicitly stored in PhaseTracer, from which we derive the symmetric partners.
 * @param reflectionIndices - The axes in field space to reflect about for this phase. Note that this should be a
 *		subset of the overall symmetry of the potential. For instance, if the potential V(x0, x1) is symmetric in both
 *		field directions, then symmetryIndices should be {0}, {1}, and {0, 1} (in separate function calls) to handle
 *		all symmetric partners.
 * @param key - The key of the new phase.
 *
 * @return A new PhaseTracer::Phase identical to the input phase with the X property negated along the axes defined in
 *		reflectionIndices, and with the input key.
 */
PhaseTracer::Phase constructSymmetricPartnerPhase(const PhaseTracer::Phase &phase, const std::vector<int> &reflectionIndices, int key);

PhaseTracer::Transition constructSymmetricPartnerTransition(const PhaseTracer::Transition &transition, const PhaseTracer::Phase &false_phase, const PhaseTracer::Phase &true_phase, const Eigen::VectorXd &false_vacuum,
                                                            const Eigen::VectorXd &true_vacuum);

/**
 *	Generates all combinations of the symmetryIndices list, stored in a list of lists. O(n*N*(2^N)), where N is the
 * number of discrete symmetries and n is the number of fields.
 */
std::vector<std::vector<int>> generateReflectionLists(const std::vector<std::vector<int>> &symmetryIndices);

/** Returns the symmetric vacuum using the axes of reflection defined through reflectionIndices. */
Eigen::VectorXd getReflectedVacuum(const Eigen::VectorXd &vacuum, const std::vector<int> &reflectionIndices,
                                   bool checkRedundancy);

/** The comparison function for sorting phases by their keys. */
bool comparePhases(const PhaseTracer::Phase &a, const PhaseTracer::Phase &b);

/** The comparison function for sorting transitions by their false vacuum keys. */
bool compareTransitions(const PhaseTracer::Transition &a, const PhaseTracer::Transition &b);

// TODO: fix excessive copying?
void extractExplicitSymmetricPhasesAndTransitions(
    const std::vector<PhaseTracer::Phase> &phases,
    const std::vector<PhaseTracer::Transition> &transitions,
    const std::vector<std::vector<int>> &symmetryIndices,
    std::vector<PhaseTracer::Phase> &out_symmetrisedPhases,
    std::vector<PhaseTracer::Transition> &out_symmetrisedTransitions);

/** Extracts the phase structure data from the given phases. */
PhaseStructureData extractPhaseStructureData(const std::vector<PhaseTracer::Phase> &phases, double T_low, double T_high);

/** Extracts the phase structure data from the given phases. */
PhaseStructureData extractPhaseStructureData(const std::vector<PhaseTracer::Phase> &phases);

/**
 * Constructs a graph representation of the transitions between phases.
 *
 * @param phases - The phases to include in the graph.
 * @param transitions - The transitions between phases.
 * @param vertices - The vertices of the graph (to be populated).
 * @param T_low - The minimum temperature to consider.
 * @param T_high - The maximum temperature to consider.
 * @param verbose - Whether to print verbose output.
 */
void constructTransitionGraph(
    const std::vector<PhaseTracer::Phase> &phases,
    const std::vector<PhaseTracer::Transition> &transitions,
    std::vector<Vertex> &vertices,
    double T_low,
    double T_high,
    bool verbose = false);

/**
 * Constructs a graph representation of the transitions between phases.
 *
 * @param phases - The phases to include in the graph.
 * @param transitions - The transitions between phases.
 * @param vertices - The vertices of the graph (to be populated).
 * @param verbose - Whether to print verbose output.
 */
void constructTransitionGraph(
    const std::vector<PhaseTracer::Phase> &phases,
    const std::vector<PhaseTracer::Transition> &transitions,
    std::vector<Vertex> &vertices,
    bool verbose = false);

/**
 * Finds all possible paths through the transition graph.
 *
 * @param vertices - The vertices of the graph.
 * @param phaseStructureData - The phase structure data.
 * @param paths - The paths through the graph (to be populated).
 * @param verbose - Whether to print verbose output.
 */
void findAllPaths(const std::vector<Vertex> &vertices, const PhaseStructureData &phaseStructureData,
                  std::vector<Path> &paths, bool verbose = false);

/**
 * Extracts the phase history from the transition finder.
 *
 * @param tf - The transition finder.
 * @param known_high_t_phase - Whether the high temperature phase is known.
 * @return The phase history paths.
 */
std::vector<Path> getPhaseHistory(const PhaseTracer::TransitionFinder &tf, bool known_high_t_phase = false);

} // namespace TransitionGraph

#endif // PHASETRACER_TRANSITION_GRAPH_UTIL_HPP_
