/* OpenMP implementation and performance impact test */

#include <ml.h>
#include <omp.h>
#include <stdio.h>

void test_train(const MLDataset *dataset, MLLinearModel *linearModel, const float trainingRate, const int epochs);
void omp_test_train(const MLDataset *dataset, MLLinearModel *linearModel, const float trainingRate, const int epochs);

int main(void)
{
    double t1 = 0.0, t2 = 0.0;

    MLLinearModel *linearModel1 = mlNewLinearModel(1.0f, 3.0f);
    MLDataset *dataset1 =
        mlNewDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 100);

    MLLinearModel *linearModel2 = mlNewLinearModel(1.0f, 3.0f);
    MLDataset *dataset2 = mlNewDataset("tests/datasets/ecommerce_user_behavior_8000.csv", "age", "time_on_site", 8000);

    puts("|| **************************************************************** ||");
    puts("|| *                  SINGLE CORE MODEL TRAINING                  * ||");
    puts("|| **************************************************************** ||\n");

    {
        t1 = omp_get_wtime();

        test_train(dataset1, linearModel1, 0.0001f, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", mlGetDatasetSize(dataset1));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());

        // ******************************** //

        t1 = omp_get_wtime();

        test_train(dataset2, linearModel2, 0.0001f, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", mlGetDatasetSize(dataset2));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());
    }

    puts("|| **************************************************************** ||");
    puts("|| *                   PARALLALEL MODEL TRAINING                  * ||");
    puts("|| **************************************************************** ||\n");

    {
        t1 = omp_get_wtime();

        omp_test_train(dataset1, linearModel1, 0.0001f, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", mlGetDatasetSize(dataset1));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());

        // ******************************** //

        t1 = omp_get_wtime();

        omp_test_train(dataset2, linearModel2, 0.0001f, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", mlGetDatasetSize(dataset2));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());
    }

    mlDeleteLinearModel(&linearModel1);
    mlDeleteLinearModel(&linearModel2);

    mlDeleteDataset(&dataset1);
    mlDeleteDataset(&dataset2);

    return 0;
}

void test_train(const MLDataset *dataset, MLLinearModel *linearModel, const float trainingRate, const int epochs)
{
    const int size = mlGetDatasetSize(dataset);

    const float *paramX = mlGetDatasetParamX(dataset);
    const float *paramY = mlGetDatasetParamY(dataset);

    float p0 = mlGetLinearModelParam0(linearModel);
    float p1 = mlGetLinearModelParam1(linearModel);

    const float k = 2.0f * trainingRate / size;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float grad0 = 0.0f;
        float grad1 = 0.0f;

        for (int i = 0; i < size; i++)
        {
            const float x = paramX[i];
            const float grad = (p0 + (p1 * x)) - paramY[i];

            grad0 += grad;
            grad1 += x * grad;
        }

        p0 -= grad0;
        p1 -= grad1;
    }

    mlSetLinearModelParam0(linearModel, p0);
    mlSetLinearModelParam1(linearModel, p1);
}

void omp_test_train(const MLDataset *dataset, MLLinearModel *linearModel, const float trainingRate, const int epochs)
{
    const int size = mlGetDatasetSize(dataset);

    const float *paramX = mlGetDatasetParamX(dataset);
    const float *paramY = mlGetDatasetParamY(dataset);

    float p0 = mlGetLinearModelParam0(linearModel);
    float p1 = mlGetLinearModelParam1(linearModel);

    const float k = 2.0f * trainingRate / size;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float grad0 = 0.0f;
        float grad1 = 0.0f;

#pragma omp parallel for reduction(+ : grad0, grad1)
        for (int i = 0; i < size; i++)
        {
            const float x = paramX[i];
            const float grad = (p0 + (p1 * x)) - paramY[i];

            grad0 += grad;
            grad1 += x * grad;
        }

        p0 -= grad0;
        p1 -= grad1;
    }

    mlSetLinearModelParam0(linearModel, p0);
    mlSetLinearModelParam1(linearModel, p1);
}