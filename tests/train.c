// ================================================================ //
// Training test //
// ================================================================ //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    LinearModel *linearModel = newLM(1.0f, 3.0f);
    Dataset *dataset = newDataset("datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    printf("Initial parameters: %.3f, %.3f\n\n", lm_getParam_0(linearModel), lm_getParam_1(linearModel));

    train(dataset, linearModel, 0.0001f, 10000);
    putchar('\n');

    printf("Final parameters: %.3f, %.3f\n", lm_getParam_0(linearModel), lm_getParam_1(linearModel));

    deleteLM(&linearModel);
    deleteDataset(&dataset);

    return 0;
}