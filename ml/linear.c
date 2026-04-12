#include <ml.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define OMP_THRESHOLD 1024

// **************************************************************** //
// * Linear Model * //
// **************************************************************** //

typedef struct LinearModel
{
    float param_0, param_1;
} LinearModel;

// **************************************************************** //

LinearModel *newLM(const float param_0, const float param_1)
{
    LinearModel *linear_model = (LinearModel *)malloc(sizeof(LinearModel));

    if (linear_model != NULL)
    {
        linear_model->param_0 = param_0;
        linear_model->param_1 = param_1;

        return linear_model;
    }

    return NULL;
}

void deleteLM(LinearModel **linear_model)
{
    if (*linear_model != NULL)
    {
        free(*linear_model);
        *linear_model = NULL;
    }
}

void lm_setParam_0(LinearModel *linear_model, const float param_0)
{
    linear_model->param_0 = param_0;
}

float lm_getParam_0(const LinearModel *linear_model)
{
    return linear_model->param_0;
}

void lm_setParam_1(LinearModel *linear_model, const float param_1)
{
    linear_model->param_1 = param_1;
}

float lm_getParam_1(const LinearModel *linear_model)
{
    return linear_model->param_1;
}

// **************************************************************** //

float predict(const LinearModel *linear_model, const float x)
{
    return (linear_model->param_0 + (linear_model->param_1 * x));
}

inline static float PREDICT(const LinearModel *linear_model, const float x)
{
    return (linear_model->param_0 + (linear_model->param_1 * x));
}

float mse(const Dataset *dataset, const LinearModel *linear_model)
{
    float mean_squared_error = 0.0f;

    const int samples = dataset_getSamples(dataset);

    const float *param_x = dataset_getParam_x(dataset);
    const float *param_y = dataset_getParam_y(dataset);

#pragma omp parallel for reduction(+ : mean_squared_error) if (samples > OMP_THRESHOLD)
    for (int i = 0; i < samples; i++)
    {
        const float e = param_y[i] - PREDICT(linear_model, param_x[i]);
        mean_squared_error += e * e;
    }

    return mean_squared_error / samples;
}

void train(const Dataset *dataset, LinearModel *linear_model, const float training_rate, const int epochs)
{
    if ((linear_model == NULL) || (dataset == NULL) || (dataset_getSamples(dataset) == 0))
    {
        return;
    }

    const int samples = dataset_getSamples(dataset);

    const float *param_x = dataset_getParam_x(dataset);
    const float *param_y = dataset_getParam_y(dataset);

    const float alpha = training_rate / samples;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float sum_grad_0 = 0.0f;
        float sum_grad_1 = 0.0f;

#pragma omp parallel for reduction(+ : sum_grad_0, sum_grad_1) if (samples > OMP_THRESHOLD)
        for (int i = 0; i < samples; i++)
        {
            const float prediction = PREDICT(linear_model, param_x[i]);
            const float grad_0 = 2.0f * (prediction - param_y[i]);

            sum_grad_0 += grad_0;
            sum_grad_1 += param_x[i] * grad_0;
        }

        linear_model->param_0 -= alpha * sum_grad_0;
        linear_model->param_1 -= alpha * sum_grad_1;
    }
}