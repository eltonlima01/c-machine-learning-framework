#include "ml.h"

#include <stdlib.h>

typedef struct LinearModel
{
    float param_0, param_1;
} LinearModel;

LinearModel *newLM(const float param_0, const float param_1)
{
    LinearModel *linearModel = (LinearModel *)malloc(sizeof(LinearModel));

    if (linearModel != NULL)
    {
        linearModel->param_0 = param_0;
        linearModel->param_1 = param_1;

        return linearModel;
    }

    return NULL;
}

void deleteLM(LinearModel **linearModel)
{
    if (*linearModel != NULL)
    {
        free(*linearModel);
        *linearModel = NULL;
    }
}

float predict(const LinearModel *linearModel, const float x)
{
    return (linearModel->param_0 + (linearModel->param_1 * x));
}

float getParam_0(const LinearModel *linearModel)
{
    return linearModel->param_0;
}

float getParam_1(const LinearModel *linearModel)
{
    return linearModel->param_1;
}