// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of an ordered tree with 4 children
// 19. March 2015

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include "BarnesHutTree.hpp"


std::ostream& operator<< (std::ostream & os, CenterOfMass const& obj) {
    os << "{ " << obj.x << " " << obj.y << " m=" << obj.mass << " }";
    return os;
}


bool TreeNode::hasChildren() const {
    return !(children[0]==nullptr && children[1]==nullptr &&
       children[2]==nullptr && children[3]==nullptr);
}


BarnesHutTree::BarnesHutTree(Box const& box)
: sqThresholdSoverD_(0.5*0.5), root_(new TreeNode(box)) {
}


void BarnesHutTree::InsertPlanet(CenterOfMass const& planet) {
    InsertPlanet(planet, root_);
}

std::pair<double, double> BarnesHutTree::TotalCenterofMass() const {
    return std::make_pair(root_->centerOfMass.x, root_->centerOfMass.y);
}

bool TreeNode::isEmpty() const {
    return (!hasChildren()) && centerOfMass.x == 0.0 && centerOfMass.y == 0.0 && centerOfMass.mass == 0.0;
}

void BarnesHutTree::InsertPlanet(CenterOfMass const& planet, std::unique_ptr<TreeNode> & node) {
    assert(IsInBox(node->box, planet));
    if(node->isEmpty()) {
        //
        // TODO implement me
        //
    }
    else if(!node->hasChildren()) {
        // Node not empty but has no children -> is a planet

        //
        // TODO implement me
        //
    }
    else {
        //
        // TODO implement me
        //
    }
}

Force BarnesHutTree::GetTotalForce(CenterOfMass const& planet) const {
    return ( GetTotalForce(planet,root_->children[0]) +
            GetTotalForce(planet,root_->children[1]) +
            GetTotalForce(planet,root_->children[2]) +
            GetTotalForce(planet,root_->children[3]) );
}

Force BarnesHutTree::GetTotalForce(CenterOfMass const& planet, std::unique_ptr<TreeNode> const& node) const {
        //
        // TODO implement me
        //
}

//++++++++++++++++++++++++++++++++ HELPER FUNCTIONS +++++++++++++++++++++++++++++++++++

Force GetForce(CenterOfMass const& obj1, CenterOfMass const& obj2) {
        //
        // TODO implement me
        //
}

bool IsInBox(Box const& box, CenterOfMass const& planet) {
    return (planet.x >= box.x0 && planet.x < box.x1 &&
       planet.y >= box.y0 && planet.y < box.y1);
}

int GetQuadrantOfPlanet(Box const& box, CenterOfMass const& planet) {
    if(planet.y >= (box.y1 + box.y0)/2.) {
        if(planet.x >= (box.x1 + box.x0)/2.) { // Top Right
            return 0;
        } else return 1; //Top Left
    }
    else {
        if (planet.x >= (box.x1 + box.x0)/2.) { //Bottom Right
            return 3;
        } else return 2; //Bottom Left
    }
}


std::array<Box,4> GetQuadrants(Box const& box) {
    std::array<Box,4> quadrants;
    double centerX = (box.x1 + box.x0) / 2.;
    double centerY = (box.y1 + box.y0) / 2.;
    Box quadrant0, quadrant1, quadrant2, quadrant3;
    quadrant0.x0 = centerX;
    quadrant0.x1 = box.x1;
    quadrant0.y0 = centerY;
    quadrant0.y1 = box.y1;
    quadrant1.x0 = box.x0;
    quadrant1.x1 = centerX;
    quadrant1.y0 = centerY;
    quadrant1.y1 = box.y1;
    quadrant2.x0 = box.x0;
    quadrant2.x1 = centerX;
    quadrant2.y0 = box.y0;
    quadrant2.y1 = centerY;
    quadrant3.x0 = centerX;
    quadrant3.x1 = box.x1;
    quadrant3.y0 = box.y0;
    quadrant3.y1 = centerY;
    quadrants[0] = quadrant0;
    quadrants[1] = quadrant1;
    quadrants[2] = quadrant2;
    quadrants[3] = quadrant3;
    return quadrants;
}


double GetDistance2(CenterOfMass const& a, CenterOfMass const& b) {
    return (a.x - b.x) * (a.x - b.x)
        + (a.y - b.y) * (a.y - b.y);
}
