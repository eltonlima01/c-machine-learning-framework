// **************************************************************** //
// * Model training test * //
// **************************************************************** //

#include <ml.hpp>
#include <iostream>

using namespace ML;
using namespace std;

int main(void)
{
    LinearModel model{1.0f, 3.0f};
    Dataset dataset{"tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours"};

    cout << "Initial parameters: " << model.getParam0() << ", " << model.getParam1() << "\n";
    cout << "Samples: " << dataset.getDatasetSize() << "\n";

    model.train(dataset, 0.0001f, 10000);

    cout << "Final parameters: " << model.getParam0() << ", " << model.getParam1() << endl;
}