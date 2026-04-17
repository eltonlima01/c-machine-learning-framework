/* Dataset loading test */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLLinearModel *linearModel = mlNewLinearModel(1.0f, 3.0f);
    MLDataset *dataset = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    puts("[Predicting Daily Usage Hours, based on Age]\n");

    printf("Linear model params: %.2f, %.2f\n\n", mlGetLinearModelParam0(linearModel), mlGetLinearModelParam1(linearModel));

    printf("3rd ocorrence: %.2f (Age) -> %.2f (Daily Usage Hours)\n", mlGetDatasetParamX(dataset),
           mlGetDatasetParamY(dataset));

    printf("3rd ocorrence (prediction): %.2f + (%.2f)x%.2f (Age) -> %.2f (Daily Usage Hours)\n",
           mlGetLinearModelParam0(linearModel), mlGetLinearModelParam1(linearModel), mlGetDatasetParamX(dataset)[2],
           mlPredict(linearModel, mlGetDatasetParamX(dataset)[2]));

    mlDeleteLinearModel(&linearModel);
    mlDeleteDataset(&dataset);

    return 0;
}