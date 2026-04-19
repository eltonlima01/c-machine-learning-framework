#pragma once

#ifdef __cplusplus
#define ML_RESTRICT __restrict

extern "C"
{
#else
#define ML_RESTRICT restrict
#endif

    void train(float *ML_RESTRICT param0, float *ML_RESTRICT param1, const float *ML_RESTRICT paramX,
               const float *ML_RESTRICT paramY, const int samples, const float training_rate, const int epochs);

#ifdef __cplusplus
}
#endif