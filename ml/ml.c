#include "ml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct Dataset
{
    float *param_x;
    float *param_y;
    int samples;
} Dataset;

Dataset *newDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples)
{
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));

    if (dataset == NULL)
    {
        return NULL;
    }

    dataset->param_x = (float *)malloc(samples * sizeof(float));
    dataset->param_y = (float *)malloc(samples * sizeof(float));
    dataset->samples = 0;

    if ((dataset->param_x == NULL) || (dataset->param_y == NULL))
    {
        if (dataset->param_x != NULL)
        {
            free(dataset->param_x);
        }

        if (dataset->param_y != NULL)
        {
            free(dataset->param_y);
        }

        free(dataset);
        return NULL;
    }

    FILE *file = fopen(datasetPath, "r");

    if (file == NULL)
    {
        deleteDataset(&dataset);
        return NULL;
    }

    char buffer[1024];

    if (fgets(buffer, sizeof(buffer), file) == NULL)
    {
        deleteDataset(&dataset);
        fclose(file);
        return NULL;
    }

    buffer[strcspn(buffer, "\r\n")] = 0;
    int index_x = -1;
    int index_y = -1;
    int current_column = 0;

    char *token = strtok(buffer, ",");
    while (token != NULL)
    {
        if (strcmp(token, param_x) == 0)
        {
            index_x = current_column;
        }

        if (strcmp(token, param_y) == 0)
        {
            index_y = current_column;
        }

        current_column++;
        token = strtok(NULL, ",");
    }

    if ((index_x == -1) || (index_y == -1))
    {
        deleteDataset(&dataset);
        fclose(file);
        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), file) && dataset->samples < samples)
    {
        current_column = 0;

        float f_x = 0.0f;
        float f_y = 0.0f;

        token = strtok(buffer, ",");

        while (token != NULL)
        {
            if (current_column == index_x)
            {
                f_x = atof(token);
            }

            if (current_column == index_y)
            {
                f_y = atof(token);
            }

            current_column++;

            token = strtok(NULL, ",");
        }

        dataset->param_x[dataset->samples] = f_x;
        dataset->param_y[dataset->samples] = f_y;
        dataset->samples++;
    }

    fclose(file);
    return dataset;
}

void deleteDataset(Dataset **dataset)
{
    if (*dataset != NULL)
    {
        free((*dataset)->param_x);
        free((*dataset)->param_y);

        free(*dataset);
        *dataset = NULL;
    }
}

float getParam_x(const Dataset *dataset, int index)
{
    return dataset->param_x[index];
}

float getParam_y(const Dataset *dataset, int index)
{
    return dataset->param_y[index];
}