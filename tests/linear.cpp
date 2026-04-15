// ****************************************************************
// * Linear regression model & prediction tests
// ****************************************************************

#include <ml.hpp>
#include <iostream>

int main ()
{
    ML::LinearModel model (1.0f, 3.0f);

    std::cout << "Params: " << model.getParam0() << ", " << model.getParam1() << "\n";
    std::cout << "Prediction (2): " << model.predict(2.0f) << std::endl;
}