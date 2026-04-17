/* ******************* */
/* Model training test */
/* ******************* */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLModel *model1 = mlNewLinear(1.0f, 3.0f);
    MLModel *model2 = mlNewLogistic(1.0f, 3.0f);

    MLDataset *dataset1 = mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 100);
    MLDataset *dataset2 = mlNewDataset("tests/datasets/ecommerce_user_behavior_8000.csv", "pages_viewed", "returning_user", 1000);

    float param0, param1;
    mlGetModelParams(model1, &param0, &param1);

    /* Linear training */

    printf("Initial parameters (Linear): %.3f, %.3f\n", param0, param1);

    mlTrain(dataset1, model1, 0.0001f, 10000);

    mlGetModelParams(model1, &param0, &param1);
    printf("Final parameters (Linear): %.3f, %.3f\n\n", param0, param1);

    /* Logistic training */

    mlGetModelParams(model2, &param0, &param1);

    printf("Initial parameters (Logistic): %.3f, %.3f\n", param0, param1);
    mlTrain(dataset2, model2, 0.0001f, 10000);

    mlGetModelParams(model2, &param0, &param1);
    printf("Final parameters (Logistic): %.3f, %.3f\n\n", param0, param1);

    mlDeleteModel(&model1);
    mlDeleteModel(&model2);
    
    mlDeleteDataset(&dataset1);
    mlDeleteDataset(&dataset2);

    return 0;
}