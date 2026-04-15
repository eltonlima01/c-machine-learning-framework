#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    void train(float *param0, float *param1, const float *paramX, const float *paramY, const int samples,
               const float training_rate, const int epochs);

#ifdef __cplusplus
}
#endif