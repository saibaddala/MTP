#include <Core>
#include <SpecialFunctions>
#include <iostream>
int main()
{
  Eigen::Array4d v(0.5,10,0,-1);
  std::cout << v.lgamma() << std::endl;
}
