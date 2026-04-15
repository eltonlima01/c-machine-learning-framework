// ****************************************************************
// * Dataset loading test
// ****************************************************************

#include <iostream>
#include <ml.hpp>

using namespace ML;
using namespace std;

int main(void)
{
    Dataset dataset{"tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours"};
    LinearModel model{1.0f, 3.0f};

    const float x = dataset.getParamXData()[3];
    const float y = dataset.getParamYData()[3];

    cout << "[Predicting Daily Usage Hours, based on Age]\n";
    cout << "Linear model params: " << model.getParam0() << ", " << model.getParam1() << "\n\n";
    cout << "[3rd ocorrence]\n";
    cout << "Age: " << x << "\nDaily Usage Hours: " << y << "\n";
    cout << "Prediction: " << model.predict(x) << "\nReal: " << y << endl;
}