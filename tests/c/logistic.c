/* ******************************************** */
/* Logistic regression model & prediction tests */
/* ******************************************** */

#include <ml.h>
#include <stdio.h>

int main(void)
{
    MLModel *model = mlNewLinear(1.0f, 3.0f);

    float param0, param1;
    mlGetModelParams(model, &param0, &param1);

    printf("Model parameters: %.3f, %.3f\n", param0, param1);
    printf("Prediction for 2: %.d\n\n", mlSigmoidClassification(model, 2.0f));

    mlSetModelParams(model, 3.0f, 1.0f);
    mlGetModelParams(model, &param0, &param1);

    printf("New parameters: %.3f, %.3f\n", param0, param1);
    printf("New prediction for 2: %.d\n", mlSigmoidClassification(model, 2.0f));

    mlDeleteModel(&model);

    return 0;
}