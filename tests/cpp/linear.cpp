// ****************************************************************
// * Linear regression model & prediction tests
// ****************************************************************

#include <iostream>
#include <ml.hpp>

using namespace ML;
using namespace std;

int main()
{
    LinearModel model(1.0f, 3.0f);

    cout << "Params: " << model.getParam0() << ", " << model.getParam1() << "\n";
    cout << "Prediction (2): " << model.predict(2.0f) << endl;
}