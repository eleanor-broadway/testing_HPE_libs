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
        //Empty Node of Tree -> insert planet
        node->centerOfMass.x = planet.x;
        node->centerOfMass.y = planet.y;
        node->centerOfMass.mass = planet.mass;
    }
    else if(!node->hasChildren()) {
        //Node contains already one planet, which needs to be pushed down to children
        //And new planet also needs to go to children
        //Finally update center of mass of this node
        CenterOfMass oldPlanet = node->centerOfMass;
        int quadrantOldPlanet = GetQuadrantOfPlanet(node->box, oldPlanet);
        if(node->children[quadrantOldPlanet]==nullptr) {
            //Create child and set it's box
            node->children[quadrantOldPlanet].reset(new TreeNode());
            node->children[quadrantOldPlanet]->box = GetQuadrants(node->box)[quadrantOldPlanet];
        }
        InsertPlanet(oldPlanet, node->children[quadrantOldPlanet]);
        int quadrantPlanet = GetQuadrantOfPlanet(node->box, planet);
        if(node->children[quadrantPlanet]==nullptr) {
            //Create child and set it's box
            node->children[quadrantPlanet].reset(new TreeNode());
            node->children[quadrantPlanet]->box = GetQuadrants(node->box)[quadrantPlanet];
        }
        InsertPlanet(planet, node->children[quadrantPlanet]);
        //Update CenterOfMass of this node
        node->centerOfMass.mass = oldPlanet.mass + planet.mass;
        node->centerOfMass.x = (oldPlanet.x * oldPlanet.mass + planet.x * planet.mass) /
                                node->centerOfMass.mass;
        node->centerOfMass.y = (oldPlanet.y * oldPlanet.mass + planet.y * planet.mass) /
                                node->centerOfMass.mass;
    }
    else {
        //Node has already children and is a center of mass not a planet
        //Push planet to children and update center of mass
        int quadrantPlanet = GetQuadrantOfPlanet(node->box, planet);
        if(node->children[quadrantPlanet]==nullptr) {
            //Create child and set it's box
            node->children[quadrantPlanet].reset(new TreeNode());
            node->children[quadrantPlanet]->box = GetQuadrants(node->box)[quadrantPlanet];
        }
        InsertPlanet(planet, node->children[quadrantPlanet]);
        node->centerOfMass.x = (node->centerOfMass.x * node->centerOfMass.mass +
                                planet.x * planet.mass) /
                                    (node->centerOfMass.mass + planet.mass);
        node->centerOfMass.y = (node->centerOfMass.y * node->centerOfMass.mass +
                                planet.y * planet.mass) /
                                    (node->centerOfMass.mass + planet.mass);
        node->centerOfMass.mass += planet.mass;
    }
}

Force BarnesHutTree::GetTotalForce(CenterOfMass const& planet) const {
    return ( GetTotalForce(planet,root_->children[0]) +
            GetTotalForce(planet,root_->children[1]) +
            GetTotalForce(planet,root_->children[2]) +
            GetTotalForce(planet,root_->children[3]) );
}

Force BarnesHutTree::GetTotalForce(CenterOfMass const& planet, std::unique_ptr<TreeNode> const& node) const {
    if(node==nullptr) {return {0.,0.};}
    if(node->centerOfMass.x == planet.x && node->centerOfMass.y == planet.y &&
       node->centerOfMass.mass == planet.mass) {return {0.,0.};}
    //if center of mass is far enough away, then return force between planet and center of mass
    double const sqBoxLength = (node->box.x1 - node->box.x0) * (node->box.y1 - node->box.y0);
    double const sqDistancePlanetNode = GetDistance2(planet, node->centerOfMass);
    if(sqBoxLength/sqDistancePlanetNode < sqThresholdSoverD_ || !node->hasChildren()) {
        return GetForce(planet, node->centerOfMass);
    }
    else {
        //if body is not far enough away, then consider sub quadrants
        return (GetTotalForce(planet, node->children[0]) +
                GetTotalForce(planet, node->children[1]) +
                GetTotalForce(planet, node->children[2]) +
                GetTotalForce(planet, node->children[3]) );
    }
}

//++++++++++++++++++++++++++++++++ HELPER FUNCTIONS +++++++++++++++++++++++++++++++++++

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
