#include <ml.h>

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define OMP_THRESHOLD 1024

inline static float SIGMOID(const float param);
inline static float PREDICT(const float param0, const float param1, const float x);

/* **************************************************************** */
/*             Model struct definition & basic functions            */
/* **************************************************************** */

typedef struct MLModel
{
    float param0, param1;
    RegressionType regressionType;
} MLModel;

MLModel *mlNewModel(RegressionType regressionType, const float param0, const float param1)
{
    if ((regressionType == ML_LINEAR) || (regressionType == ML_LOGISTIC))
    {
        MLModel *model = (MLModel *)malloc(sizeof(MLModel));

        if (model != NULL)
        {
            model->regressionType = regressionType;

            model->param0 = param0;
            model->param1 = param1;

            return model;
        }
    }

    return NULL;
}

void mlDeleteModel(MLModel **model)
{
    if ((model != NULL) && (*model != NULL))
    {
        free(*model);
        *model = NULL;
    }
}

void mlGetModelParams(const MLModel *model, float *param0, float *param1)
{
    *param0 = model->param0;
    *param1 = model->param1;
}

void mlSetModelParams(MLModel *model, const float param0, const float param1)
{
    model->param0 = param0;
    model->param1 = param1;
}

/* *************** */
/* Basic functions */
/* *************** */

float mlPredict(const MLModel *model, const float param)
{
    return (model->regressionType == ML_LINEAR) ? (model->param0 + (model->param1 * param))
                                                : SIGMOID(model->param0 + (model->param1 * param));
}

float mlMSE(const MLModel *restrict model, const MLDataset *restrict dataset)
{
    const int size = mlGetDatasetSize(dataset);

    const float *restrict paramX = mlGetDatasetParamXData(dataset);
    const float *restrict paramY = mlGetDatasetParamYData(dataset);

    const float param0 = model->param0;
    const float param1 = model->param1;

    float meanSquaredError = 0.0f;

    if (model->regressionType == ML_LINEAR)
    {
#pragma omp parallel for simd reduction(+ : meanSquaredError) if (size > OMP_THRESHOLD)
        for (int i = 0; i < size; i++)
        {
            const float e = paramY[i] - PREDICT(param0, param1, paramX[i]);
            meanSquaredError += e * e;
        }
    }
    else
    {
#pragma omp parallel for simd reduction(+ : meanSquaredError) if (size > OMP_THRESHOLD)
        for (int i = 0; i < size; i++)
        {
            const float e = paramY[i] - SIGMOID(param0 + (param1 * paramX[i]));
            meanSquaredError += e * e;
        }
    }

    return meanSquaredError / size;
}

void mlTrainModel(MLModel *restrict model, const MLDataset *restrict dataset, const float trainingRate,
                  const int epochs)
{
    if ((model == NULL) || (dataset == NULL) || (mlGetDatasetSize(dataset) == 0))
    {
        return;
    }

    const int datasetSize = mlGetDatasetSize(dataset);

    float param0 = model->param0;
    float param1 = model->param1;

    const float *restrict paramX = mlGetDatasetParamXData(dataset);
    const float *restrict paramY = mlGetDatasetParamYData(dataset);

    if (model->regressionType == ML_LINEAR)
    {
        const float k = 2.0f * trainingRate / datasetSize;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float grad0 = 0.0f;
            float grad1 = 0.0f;

#pragma omp parallel for simd reduction(+ : grad0, grad1) if (datasetSize > OMP_THRESHOLD)
            for (int i = 0; i < datasetSize; i++)
            {
                const float x = paramX[i];
                const float grad = PREDICT(param0, param1, x) - paramY[i];

                grad0 += grad;
                grad1 += x * grad;
            }

            param0 -= k * grad0;
            param1 -= k * grad1;
        }
    }
    else
    {
        const float k = trainingRate / datasetSize;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float grad0 = 0.0f;
            float grad1 = 0.0f;

#pragma omp parallel for simd reduction(+ : grad0, grad1) if (datasetSize > OMP_THRESHOLD)
            for (int i = 0; i < datasetSize; i++)
            {
                const float x = paramX[i];
                const float grad = SIGMOID(param0 + (param1 * x)) - paramY[i];

                grad0 += grad;
                grad1 += x * grad;
            }

            param0 -= k * grad0;
            param1 -= k * grad1;
        }
    }

    model->param0 = param0;
    model->param1 = param1;
}

/* Inline */

inline static float SIGMOID(const float param)
{
    return (1.0f / (1.0f + expf(-param)));
}

inline static float PREDICT(const float param0, const float param1, const float param)
{
    return (param0 + (param1 * param));
}

#undef OMP_THRESHOLD