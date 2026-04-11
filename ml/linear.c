#include <ml.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

inline static float PREDICT(const LinearModel *linearModel, const float x);

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
    const int samples = dataset_getSamples(dataset);

#pragma omp parallel for reduction(+ : error)
    for (int i = 0; i < samples; i++)
    {
        const float e = dataset_getParam_y(dataset, i) - PREDICT(linearModel, dataset_getParam_x(dataset, i));
        error += e * e;
    }

    return error / samples;
}

void train(const Dataset *dataset, LinearModel *linearModel, float trainingRate, int epochs)
{
    if ((linearModel == NULL) || (dataset == NULL) || (dataset_getSamples(dataset) == 0))
    {
        return;
    }

    const int samples = dataset_getSamples(dataset);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float sumGrad_0 = 0.0f;
        float sumGrad_1 = 0.0f;

#pragma omp parallel for reduction(+ : sumGrad_0, sumGrad_1)
        for (int i = 0; i < samples; i++)
        {
            const float param_x = dataset_getParam_x(dataset, i);
            const float param_y = dataset_getParam_y(dataset, i);

            const float error = PREDICT(linearModel, param_x);

            sumGrad_0 += 2.0f * (error - param_y);
            sumGrad_1 += 2.0f * param_x * (error - param_y);
        }

        linearModel->param_0 -= trainingRate * (sumGrad_0 / samples);
        linearModel->param_1 -= trainingRate * (sumGrad_1 / samples);
    }
}

// ================================================================ //

inline static float PREDICT(const LinearModel *linearModel, const float x)
{
    return (linearModel->param_0 + (linearModel->param_1 * x));
}