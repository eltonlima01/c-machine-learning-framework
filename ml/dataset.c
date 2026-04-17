#include <ml.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Dataset */

typedef struct MLDataset
{
    int size;
    float *paramX;
    float *paramY;
} MLDataset;

// **************************************************************** //

MLDataset *mlNewDataset(const char *datasetPath, const char *paramX, const char *paramY, const int size)
{
    MLDataset *dataset = (MLDataset *)malloc(sizeof(MLDataset));

    if (dataset == NULL)
    {
        return NULL;
    }

    dataset->size = 0;

    dataset->paramX = (float *)malloc(size * sizeof(float));

    if (dataset->paramX == NULL)
    {
        free(dataset);
        return NULL;
    }

    dataset->paramY = (float *)malloc(size * sizeof(float));

    if (dataset->paramY == NULL)
    {
        free(dataset->paramX);
        free(dataset);

        return NULL;
    }

    FILE *file = fopen(datasetPath, "r");

    if (file == NULL)
    {
        mlDeleteDataset(&dataset);
        return NULL;
    }

    char buffer[1024];

    if (fgets(buffer, sizeof(buffer), file) == NULL)
    {
        mlDeleteDataset(&dataset);
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
        if (strcmp(token, paramX) == 0)
        {
            index_x = current_column;
        }

        if (strcmp(token, paramY) == 0)
        {
            index_y = current_column;
        }

        current_column++;
        token = strtok(NULL, ",");
    }

    if ((index_x == -1) || (index_y == -1))
    {
        mlDeleteDataset(&dataset);
        fclose(file);

        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), file) && dataset->size < size)
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

        dataset->paramX[dataset->size] = f_x;
        dataset->paramY[dataset->size] = f_y;
        dataset->size++;
    }

    fclose(file);
    return dataset;
}

void mlDeleteDataset(MLDataset **dataset)
{
    if (*dataset != NULL)
    {
        free((*dataset)->paramX);
        free((*dataset)->paramY);

        free(*dataset);
        *dataset = NULL;
    }
}

/* **************************************************************** */

const float *mlGetDatasetParamX(const MLDataset *dataset)
{
    return dataset->paramX;
}

const float *mlGetDatasetParamY(const MLDataset *dataset)
{
    return dataset->paramY;
}

int mlGetDatasetSize(const MLDataset *dataset)
{
    return dataset->size;
}