// **************************************************************** //
// * OpenMP implementation and performance impact test * //
// **************************************************************** //

#include <ml.h>
#include <omp.h>
#include <stdio.h>

void test_train(const Dataset *dataset, LinearModel *linear_model, const int epochs);
void omp_test_train(const Dataset *dataset, LinearModel *linear_model, const int epochs);

int main(void)
{
    double t1 = 0.0, t2 = 0.0;

    LinearModel *linear_model_1 = newLM(1.0f, 3.0f);
    Dataset *dataset_1 =
        newDataset("tests/datasets/AI_Student_Life_Pakistan_2026.csv", "Age", "Daily_Usage_Hours", 100);

    LinearModel *linear_model_2 = newLM(1.0f, 3.0f);
    Dataset *dataset_2 = newDataset("tests/datasets/ecommerce_user_behavior_8000.csv", "age", "time_on_site", 8000);

    puts("|| **************************************************************** ||");
    puts("|| * SINGLE CORE MEAN SQUARED ERROR CALCULATION * ||");
    puts("|| **************************************************************** ||\n");

    {
        t1 = omp_get_wtime();

        test_train(dataset_1, linear_model_1, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", dataset_getSamples(dataset_1));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());

        // ******************************** //

        t1 = omp_get_wtime();

        test_train(dataset_2, linear_model_2, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", dataset_getSamples(dataset_2));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());
    }

    puts("|| **************************************************************** ||");
    puts("|| * PARALLALEL MEAN SQUARED ERROR CALCULATION *||");
    puts("|| **************************************************************** ||\n");

    {
        t1 = omp_get_wtime();

        omp_test_train(dataset_1, linear_model_1, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", dataset_getSamples(dataset_1));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());

        // ******************************** //

        t1 = omp_get_wtime();

        omp_test_train(dataset_2, linear_model_2, 10000);

        t2 = omp_get_wtime();

        printf("Samples: %d\n", dataset_getSamples(dataset_2));
        printf("Execution time = %.6f\n", t2 - t1);
        printf("Precision = %.6f\n\n", omp_get_wtick());
    }

    deleteLM(&linear_model_1);
    deleteLM(&linear_model_2);

    deleteDataset(&dataset_1);
    deleteDataset(&dataset_2);

    return 0;
}

inline static float PREDICT(const LinearModel *linear_model, const float x)
{
    return (lm_getParam_0(linear_model) + (lm_getParam_1(linear_model) * x));
}

void test_train(const Dataset *dataset, LinearModel *linear_model, const int epochs)
{
    const int samples = dataset_getSamples(dataset);
    const float *param_x = dataset_getParam_x(dataset);
    const float *param_y = dataset_getParam_y(dataset);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float sum_grad_0 = 0.0f;
        float sum_grad_1 = 0.0f;

        for (int i = 0; i < samples; i++)
        {
            const float prediction = PREDICT(linear_model, param_x[i]);
            const float grad_0 = 2.0f * (prediction - param_y[i]);

            sum_grad_0 += grad_0;
            sum_grad_1 += param_x[i] * grad_0;
        }

        lm_setParam_0(linear_model, lm_getParam_0(linear_model) - (0.001f * (sum_grad_0 / samples)));
        lm_setParam_1(linear_model, lm_getParam_1(linear_model) - (0.001f * (sum_grad_1 / samples)));
    }
}

void omp_test_train(const Dataset *dataset, LinearModel *linear_model, const int epochs)
{
    const int samples = dataset_getSamples(dataset);
    const float *param_x = dataset_getParam_x(dataset);
    const float *param_y = dataset_getParam_y(dataset);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float sum_grad_0 = 0.0f;
        float sum_grad_1 = 0.0f;

#pragma omp parallel for reduction(+ : sum_grad_0, sum_grad_1)
        for (int i = 0; i < samples; i++)
        {
            const float prediction = PREDICT(linear_model, param_x[i]);
            const float grad_0 = 2.0f * (prediction - param_y[i]);

            sum_grad_0 += grad_0;
            sum_grad_1 += param_x[i] * grad_0;
        }

        lm_setParam_0(linear_model, lm_getParam_0(linear_model) - (0.001f * (sum_grad_0 / samples)));
        lm_setParam_1(linear_model, lm_getParam_1(linear_model) - (0.001f * (sum_grad_1 / samples)));
    }
}