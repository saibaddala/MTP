#include <Core>
#include <SpecialFunctions>
#include <iostream>
int main()
{
  Eigen::Array4d v(-0.5,2,0,-7);
  std::cout << v.erfc() << std::endl;
}
