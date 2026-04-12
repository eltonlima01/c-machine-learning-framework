// **************************************************************** //
// * Linear regression model & prediction tests * //
// **************************************************************** //

#include <ml.h>
#include <stdio.h>

int main(void)
{
    LinearModel *linearModel = newLM(3.75f, 6.78e-5f);

    puts("Cyprus -> PIB US$ 32.655:");

    printf("Life satisfaction = %.2f + (6,78x10⁻⁵)xPIB\n= %.2f + (6,78x10⁻⁵)x32655\n= %.2f\n",
           lm_getParam_0(linearModel), lm_getParam_0(linearModel), predict(linearModel, 32655.0f));

    deleteLM(&linearModel);

    return 0;
}