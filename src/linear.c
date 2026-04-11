#include "ml.h"

#include <stdio.h>
#include <stdlib.h>

// ================================================================ //

typedef struct LinearModel
{
    float param_0, param_1;
} LinearModel;

// ================================================================ //

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

// ================================================================ //

float predict(const LinearModel *linearModel, const float x)
{
    return (linearModel->param_0 + (linearModel->param_1 * x));
}

float lm_getParam_0(const LinearModel *linearModel)
{
    return linearModel->param_0;
}

float lm_getParam_1(const LinearModel *linearModel)
{
    return linearModel->param_1;
}

// ================================================================ //

float MSE(const Dataset *dataset, const LinearModel *linearModel)
{
    float error = 0.0f;

    for (int i = 0; i < dataset_getSamples(dataset); i++)
    {
        const float e = dataset_getParam_y(dataset, i) - predict(linearModel, dataset_getParam_x(dataset, i));
        error += e * e;
    }

    return error / dataset_getSamples(dataset);
}

void train(const Dataset *dataset, LinearModel *linearModel, float trainingRate, int epochs)
{
    if ((linearModel == NULL) || (dataset == NULL) || (dataset_getSamples(dataset) == 0))
    {
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float sumGrad_0 = 0.0f;
        float sumGrad_1 = 0.0f;

        for (int i = 0; i < dataset_getSamples(dataset); i++)
        {
            const float error = predict(linearModel, dataset_getParam_x(dataset, i));

            sumGrad_0 += 2.0f * (error - dataset_getParam_y(dataset, i));
            sumGrad_1 += 2.0f * dataset_getParam_x(dataset, i) * (error - dataset_getParam_y(dataset, i));
        }

        float grad_0 = sumGrad_0 / dataset_getSamples(dataset);
        float grad_1 = sumGrad_1 / dataset_getSamples(dataset);

        linearModel->param_0 -= trainingRate * grad_0;
        linearModel->param_1 -= trainingRate * grad_1;

        if ((epoch % 1000) == 0)
        {
            printf("[Epoch %d]\nMSE: %.3f\t-\tθ⁰ = %.3f\t-\tθ¹ = %.3f\n\n", epoch, MSE(dataset, linearModel),
                   linearModel->param_0, linearModel->param_1);
        }
    }
}