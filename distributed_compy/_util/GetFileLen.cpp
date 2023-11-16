#include "GetFileLen.h"

long GetFileLen(const char* data_file_name)
{
    long size = 0;
    FILE *fp = fopen(data_file_name, "rb");
    if (!fp)
    {
        return 0;
    }
    fseek(fp, 0, SEEK_END);    //move file pointer to end of file
    size = ftell(fp);
    rewind(fp);
    fclose(fp);
    return size/sizeof(float);
}