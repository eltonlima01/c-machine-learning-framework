// **************************************************************** //
// * Model training test * //
// **************************************************************** //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    LinearModel *linear_model = newLM(1.0f, 3.0f);
    Dataset *dataset = newDataset("datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Initial parameters: %.3f, %.3f\n\n", lm_getParam_0(linear_model), lm_getParam_1(linear_model));

    train(dataset, linear_model, 0.0001f, 10000);
    putchar('\n');

    printf("Final parameters: %.3f, %.3f\n", lm_getParam_0(linear_model), lm_getParam_1(linear_model));

    deleteLM(&linear_model);
    deleteDataset(&dataset);

    return 0;
}