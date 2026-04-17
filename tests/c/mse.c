/* *********************************** */
/* Mean Squared Error calculation test */
/* *********************************** */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLModel *model = mlNewLinear(1.0f, 3.0f);
    MLDataset *dataset = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 100);

    float param0, param1;
    mlGetModelParams(model, &param0, &param1);

    printf("Model parameters: %.3f, %.3f\n", param0, param1);
    printf("Mean squared error for Age -> Daily Usage Hours: %.3f\n", mlMSE(dataset, model));

    mlDeleteModel(&model);
    mlDeleteDataset(&dataset);

    return 0;
}