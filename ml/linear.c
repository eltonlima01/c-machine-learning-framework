#include <ml.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define OMP_THRESHOLD 1024

// **************************************************************** //
// * Linear Model * //
// **************************************************************** //

typedef struct MLLinearModel
{
    float param0, param1;
} MLLinearModel;

// **************************************************************** //

MLLinearModel *mlNewLinearModel(const float param0, const float param1)
{
    MLLinearModel *linearModel = (MLLinearModel *)malloc(sizeof(MLLinearModel));

    if (linearModel != NULL)
    {
        linearModel->param0 = param0;
        linearModel->param1 = param1;

        return linearModel;
    }

    return NULL;
}

void mlDeleteLinearModel(MLLinearModel **linearModel)
{
    if (*linearModel != NULL)
    {
        free(*linearModel);
        *linearModel = NULL;
    }
}

void mlSetLinearModelParam0(MLLinearModel *linearModel, const float param0)
{
    linearModel->param0 = param0;
}

float mlGetLinearModelParam0(const MLLinearModel *linearModel)
{
    return linearModel->param0;
}

void mlSetLinearModelParam1(MLLinearModel *linearModel, const float param1)
{
    linearModel->param1 = param1;
}

float mlGetLinearModelParam1(const MLLinearModel *linearModel)
{
    return linearModel->param1;
}

// **************************************************************** //

float mlPredict(const MLLinearModel *linearModel, const float x)
{
    return (linearModel->param0 + (linearModel->param1 * x));
}

inline static float PREDICT(const MLLinearModel *linearModel, const float x)
{
    return (linearModel->param0 + (linearModel->param1 * x));
}

float mlMSE(const MLDataset *dataset, const MLLinearModel *linearModel)
{
    const int size = mlGetDatasetSize(dataset);
    
    const float * restrict paramX = mlGetDatasetParamX(dataset);
    const float * restrict paramY = mlGetDatasetParamY(dataset);

    float meanSquaredError = 0.0f;

#pragma omp parallel for simd reduction(+ : meanSquaredError) if (size > OMP_THRESHOLD)
    for (int i = 0; i < size; i++)
    {
        const float e = paramY[i] - PREDICT(linearModel, paramX[i]);
        meanSquaredError += e * e;
    }

    return meanSquaredError / size;
}

void mlTrain(const MLDataset *dataset, MLLinearModel *linearModel, const float training_rate, const int epochs)
{
    if ((linearModel == NULL) || (dataset == NULL) || (mlGetDatasetSize(dataset) == 0))
    {
        return;
    }

    const int size = mlGetDatasetSize(dataset);

    float param0 = linearModel->param0;
    float param1 = linearModel->param1;

    const float * restrict paramX = mlGetDatasetParamX(dataset);
    const float * restrict paramY = mlGetDatasetParamY(dataset);

    const float k = 2.0f * training_rate / size;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float grad0 = 0.0f;
        float grad1 = 0.0f;

#pragma omp parallel for simd reduction(+ : grad0, grad1) if (size > OMP_THRESHOLD)
        for (int i = 0; i < size; i++)
        {
            const float x = paramX[i];
            const float grad = param0 + (param1 * x) - paramY[i];

            grad0 += grad;
            grad1 += x * grad;
        }

        param0 -= k * grad0;
        param1 -= k * grad1;
    }

    linearModel->param0 = param0;
    linearModel->param1 = param1;
}

#undef OMP_THRESHOLD