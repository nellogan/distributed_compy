#include "NaiveSum.h"

void NaiveSum(float* data, float* out, int size)
{
    float res = 0;
    for (int i=0; i<size; i++)
    {
        res+=data[i];
    }
    *out = res;
}