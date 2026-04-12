// **************************************************************** //
// * Mean Squared Error calculation test * //
// **************************************************************** //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    LinearModel *linear_model = newLM(1.0f, 3.0f);
    Dataset *dataset = newDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Linear model params: %.2f, %.2f\n", lm_getParam_0(linear_model), lm_getParam_1(linear_model));

    printf("Mean squared error: %.2f\n", mse(dataset, linear_model));

    deleteLM(&linear_model);
    deleteDataset(&dataset);

    return 0;
}