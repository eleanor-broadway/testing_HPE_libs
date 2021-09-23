// HPCSE II FS 2015, Exercise on Barnes-Hut Algorithm
// Implementation of a ordered tree with 4 children
// Copyright by Damian Steiger
// 19. March 2015
#ifndef BARNES_HUT_TREE_HPP
#define BARNES_HUT_TREE_HPP

#include <memory>
#include <array>
#include <utility>
#include <vector>


struct vector_array
{
    vector_array(unsigned int n)
    : x(n), y(n)
    {
    }

    friend void swap(vector_array & a, vector_array & b)
    {
        swap(a.x, b.x);
        swap(a.y, b.y);
    }

    std::vector<double> x;
    std::vector<double> y;
};

struct Force {
    double aX;
    double aY;
    Force & operator+=(const Force & other) {
        aX += other.aX;
        aY += other.aY;
        return *this;
    }
};

inline Force operator + (const Force & a, const Force & b) {
    Force totalForce(a);
    totalForce += b;
    return totalForce;
}

struct CenterOfMass {
    CenterOfMass() :
    x(0), y(0), mass(0)
    {}
    CenterOfMass(double x, double y, double mass)
    : x(x), y(y), mass(mass)  { }
    double x;
    double y;
    double mass;
};

std::ostream& operator<< (std::ostream &, CenterOfMass const&);

struct Box {
    double x0;
    double x1;
    double y0;
    double y1;
};

// Node of the tree which contains the pointers to the 4 children, the box size and
// center of mass
struct TreeNode {
    TreeNode()
    {
    }
    TreeNode(Box const& box)
    : box(box)
    {
    }
    // Returns true if all children are empty
    bool hasChildren() const;
    bool isEmpty() const;
    CenterOfMass centerOfMass;
    Box box;
    std::array<std::unique_ptr<TreeNode>,4> children;
};

// Implements the tree and BarnesHut Algorithm
class BarnesHutTree {
public:
    // You need to give a starting box size
    BarnesHutTree(Box const&);
    // Insert Planet into the tree. Center of mass gets updated automatically.
    // So once all planets are inserted, you are ready to calculate the force
    // Precondition: Planet needs to be contained in initial box size!
    //               Each planet must have unique position at all times!
    void InsertPlanet(CenterOfMass const&);
    // Calculate total force on CenterOfMass object (excluding himself)
    Force GetTotalForce(CenterOfMass const&) const;
    
    //Sets a new threshold
    void SetThresholdSoverD(double);
    
    std::pair<double, double> TotalCenterofMass() const;
    
private:
    double sqThresholdSoverD_;
    std::unique_ptr<TreeNode> root_;
    
    void InsertPlanet(CenterOfMass const&, std::unique_ptr<TreeNode> &);
    //calculated total force of this subtree
    Force GetTotalForce(CenterOfMass const&, std::unique_ptr<TreeNode> const&) const;
    
};

//+++++++++++++++++++++++++++++++++++++ HELPER FUNCTIONS +++++++++++++++++++++++++

//Returns True if Planet is in Box
bool IsInBox(Box const&, CenterOfMass const&);

//Returns in which Quadrant the Planet is if box is divided into 4 quadrants:
//    1 | 0
//   -------
//    2 | 3
int GetQuadrantOfPlanet(Box const&, CenterOfMass const&);

//Returns a std::array with the 4 Quadrants given a box
std::array<Box,4> GetQuadrants(Box const&);

//Get force between two objects of type CenterOfMass
Force GetForce(CenterOfMass const&,CenterOfMass const&);

double GetDistance2(CenterOfMass const&, CenterOfMass const&);


#endif // BARNES_HUT_TREE_HPP
