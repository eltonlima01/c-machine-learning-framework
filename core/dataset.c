#include <ml.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* **************************************************************** */
/*    Dataset struct definition for CSV loading & basic functions   */
/* **************************************************************** */

typedef struct MLDataset
{
    int size;
    float *paramXData;
    float *paramYData;
} MLDataset;

/* ***************************************** */
/* Dataset struct definition for CSV loading */
/* ***************************************** */

MLDataset *mlNewDataset(const char *datasetPath, const char *paramX, const char *paramY, const int size)
{
    MLDataset *dataset = (MLDataset *)malloc(sizeof(MLDataset));

    if (dataset == NULL)
    {
        return NULL;
    }

    dataset->size = 0;

    dataset->paramXData = (float *)malloc(size * sizeof(float));

    if (dataset->paramXData == NULL)
    {
        free(dataset);
        return NULL;
    }

    dataset->paramYData = (float *)malloc(size * sizeof(float));

    if (dataset->paramYData == NULL)
    {
        free(dataset->paramXData);
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

    char *p = buffer;
    while (*p != '\0')
    {
        char *token = p;

        while((*p != ',') && (*p != '\0'))
        {
            p++;
        }

        if (*p == ',')
        {
            *p = '\0';
            p++;
        }

        if (strcmp(token, paramX) == 0)
        {
            index_x = current_column;
        }

        if (strcmp(token, paramY) == 0)
        {
            index_y = current_column;
        }

        current_column++;
    }

    if ((index_x == -1) || (index_y == -1))
    {
        mlDeleteDataset(&dataset);
        fclose(file);

        return NULL;
    }

    while (fgets(buffer, sizeof(buffer), file) && dataset->size < size)
    {
        buffer [strcspn(buffer, "\r\n")] = 0;
        current_column = 0;

        float f_x = 0.0f;
        float f_y = 0.0f;

        char *p = buffer;

        while (*p != '\0')
        {
            char *token = p;

            while((*p != ',') && (*p != '\0'))
            {
                p++;
            }

            if (*p == ',')
            {
                *p = '\0';
                p++;
            }

            if (current_column == index_x)
            {
                f_x = atof(token);
            }

            if (current_column == index_y)
            {
                f_y = atof(token);
            }

            current_column++;
        }

        dataset->paramXData[dataset->size] = f_x;
        dataset->paramYData[dataset->size] = f_y;
        dataset->size++;
    }

    fclose(file);
    return dataset;
}

/* *************** */
/* Basic functions */
/* *************** */

void mlDeleteDataset(MLDataset **dataset)
{
    if ((dataset != NULL) && (*dataset != NULL))
    {
        free((*dataset)->paramXData);
        free((*dataset)->paramYData);

        free(*dataset);
        *dataset = NULL;
    }
}

const float *mlGetDatasetParamXData(const MLDataset *dataset)
{
    return dataset->paramXData;
}

const float *mlGetDatasetParamYData(const MLDataset *dataset)
{
    return dataset->paramYData;
}

int mlGetDatasetSize(const MLDataset *dataset)
{
    return dataset->size;
}