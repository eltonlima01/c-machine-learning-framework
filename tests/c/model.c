/* *********************************** */
/* Model definition & prediction tests */
/* *********************************** */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLModel *linearModel = mlNewModel(ML_LINEAR, 1.0f, 3.0f);
    MLModel *logisticModel = mlNewModel(ML_LOGISTIC, 3.0f, 1.0f);

    float param0, param1;
    mlGetModelParams(linearModel, &param0, &param1);

    printf("Linear model parameters: %.3f, %.3f\n", param0, param1);
    printf("Linear Prediction for 2: %.3f\n\n", mlPredict(linearModel, 2.0f));

    mlGetModelParams(logisticModel, &param0, &param1);

    printf("Logistic model parameters: %.3f, %.3f\n", param0, param1);
    printf("Logistic prediction for 2: %.3f\n\n", mlPredict(logisticModel, 2.0f));

    mlSetModelParams(linearModel, 3.0f, 1.0f);
    mlGetModelParams(linearModel, &param0, &param1);

    printf("New linear model parameters: %.3f, %.3f\n", param0, param1);
    printf("New linear prediction for 2: %.3f\n", mlPredict(linearModel, 2.0f));

    mlDeleteModel(&linearModel);
    mlDeleteModel(&logisticModel);

    return 0;
}