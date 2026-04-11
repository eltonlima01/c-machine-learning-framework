// ================================================================ //
// MSE calculation test //
// ================================================================ //

#include "../ml/ml.h"
#include <stdio.h>

int main(void)
{
    LinearModel *linearModel = newLM(1.0f, 3.0f);
    Dataset *dataset = newDataset("datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Linear model params: %.2f, %.2f\n", lm_getParam_0(linearModel), lm_getParam_1(linearModel));

    printf("Mean squared error: %.2f\n", MSE(dataset, linearModel));

    deleteLM(&linearModel);
    deleteDataset(&dataset);

    return 0;
}