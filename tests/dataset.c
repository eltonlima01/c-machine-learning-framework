// **************************************************************** //
// * Dataset loading test * //
// **************************************************************** //

// * FIX: dataset_getParam_y, dataset_getParam_x * //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    LinearModel *linearModel = newLM(1.0f, 3.0f);
    Dataset *dataset = newDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 20);

    puts("[Predicting Daily Usage Hours, based on Age]\n");

    printf("Linear model params: %.2f, %.2f\n\n", lm_getParam_0(linearModel), lm_getParam_1(linearModel));

    printf("3rd ocorrence: %.2f (Age) -> %.2f (Daily Usage Hours)\n", dataset_getParam_x(dataset, 3),
           dataset_getParam_y(dataset, 3));

    printf("3rd ocorrence (prediction): %.2f + (%.2f)x%.2f (Age) -> %.2f (Daily Usage Hours)\n",
           lm_getParam_0(linearModel), lm_getParam_1(linearModel), dataset_getParam_x(dataset, 3),
           predict(linearModel, dataset_getParam_x(dataset, 3)));

    deleteLM(&linearModel);
    deleteDataset(&dataset);

    return 0;
}