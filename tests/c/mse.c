/* Mean Squared Error calculation test */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLLinearModel *linearModel = mlNewLinearModel(1.0f, 3.0f);
    MLDataset *dataset = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Linear model params: %.2f, %.2f\n", mlGetLinearModelParam0(linearModel), mlGetLinearModelParam1(linearModel));
    printf("Mean squared error: %.2f\n", mlMSE(dataset, linearModel));

    mlDeleteLinearModel(&linearModel);
    mlDeleteDataset(&dataset);

    return 0;
}