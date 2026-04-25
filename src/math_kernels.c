#include <math_kernels.h>

#include <omp.h>
#include <math.h>

#define OMP_THRESHOLD 1024

inline static float PREDICT_LINEAR(const float param0, const float param1, const float param);
inline static float PREDICT_LOGISTIC(const float param);

/* **************************************************************** */
/*                      Linear training function                    */
/* **************************************************************** */

void trainLinear(float *ML_RESTRICT param0, float *ML_RESTRICT param1, const float *ML_RESTRICT paramX,
           const float *ML_RESTRICT paramY, const int samples, const float training_rate, const int epochs)
{
    const float k = 2.0f * training_rate / samples;

    float p0 = *param0;
    float p1 = *param1;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float grad0 = 0.0f;
        float grad1 = 0.0f;

#pragma omp parallel for simd reduction(+ : grad0, grad1) if (samples > OMP_THRESHOLD)
        for (int i = 0; i < samples; i++)
        {
            const float x = paramX[i];
            const float grad = PREDICT_LINEAR(p0, p1, x) - paramY[i];

            grad0 += grad;
            grad1 += x * grad;
        }

        p0 -= k * grad0;
        p1 -= k * grad1;
    }

    *param0 = p0;
    *param1 = p1;
}

/* **************************************************************** */
/*                    Logistic training function                    */
/* **************************************************************** */

void trainLogistic(float *ML_RESTRICT param0, float *ML_RESTRICT param1, const float *ML_RESTRICT paramX,
           const float *ML_RESTRICT paramY, const int samples, const float training_rate, const int epochs)
{
    const float k = training_rate / samples;

    float p0 = *param0;
    float p1 = *param1;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float grad0 = 0.0f;
        float grad1 = 0.0f;

#pragma omp parallel for simd reduction(+ : grad0, grad1) if (samples > OMP_THRESHOLD)
        for (int i = 0; i < samples; i++)
        {
            const float x = paramX[i];
            const float grad = PREDICT_LOGISTIC(PREDICT_LINEAR(p0, p1, x)) - paramY[i];

            grad0 += grad;
            grad1 += x * grad;
        }

        p0 -= k * grad0;
        p1 -= k * grad1;
    }

    *param0 = p0;
    *param1 = p1;
}

/* Inline */

inline static float PREDICT_LINEAR(const float param0, const float param1, const float param)
{
    return (param0 + (param1 * param));
}

inline static float PREDICT_LOGISTIC(const float param)
{
    return (1.0f / (1.0f + expf(-param)));
}

#undef OMP_THRESHOLD