#include "DataExtractor.h"

float* DataExtractor(const char* data_file_name, long num_elements)
{
    float* data;
    data = (float*)malloc(num_elements*sizeof(float));
    FILE *fp = fopen(data_file_name, "rb+");
    fread(data, sizeof(float), num_elements, fp);
    fclose(fp);
    return data;
}