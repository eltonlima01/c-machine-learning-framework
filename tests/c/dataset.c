/* ******************** */
/* Dataset loading test */
/* ******************** */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    const char *paramX = "Age";
    const char *paramY = "Daily_Usage_Hours";

    MLModel *model = mlNewLinear(1.0f, 3.0f);
    MLDataset *dataset = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", paramX, paramY, 100);

    float param0, param1;
    mlGetModelParams(model, &param0, &param1);

    printf("Model parameters: %.3f, %.3f\n", param0, param1);
    printf("Prediction for 3rd ocurrence of %s -> %s: %.3f\n", paramX, paramY, mlPredict(model, mlGetDatasetParamXData(dataset)[2]));

    mlDeleteModel(&model);
    mlDeleteDataset(&dataset);

    return 0;
}