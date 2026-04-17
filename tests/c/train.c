// **************************************************************** //
// * Model training test * //
// **************************************************************** //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLLinearModel *linearModel = mlNewLinearModel(1.0f, 3.0f);
    MLDataset *dataset = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Initial parameters: %.3f, %.3f\n", mlGetLinearModelParam0(linearModel), mlGetLinearModelParam1(linearModel));

    mlTrain(dataset, linearModel, 0.0001f, 10000);
    putchar('\n');

    printf("Final parameters: %.3f, %.3f\n", mlGetLinearModelParam0(linearModel), mlGetLinearModelParam1(linearModel));

    mlDeleteLinearModel(&linearModel);
    mlDeleteDataset(&dataset);

    return 0;
}