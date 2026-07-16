#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstdint>
struct vectorset
{
    float* data = nullptr;
    uint16_t* data_half = nullptr;
    size_t dim, vecnum;
    int* codes = nullptr;
    bool data_is_half = false;
    vectorset(float* _data, int _dim, int _vecnum):data(_data), dim(_dim), vecnum(_vecnum){}
    vectorset(float* _data, int* _codes, int _dim, int _vecnum): data(_data), dim(_dim), vecnum(_vecnum), codes(_codes){}
    vectorset(uint16_t* _data_half, int* _codes, int _dim, int _vecnum): data_half(_data_half), dim(_dim), vecnum(_vecnum), codes(_codes), data_is_half(true){}
    ~vectorset(){
    }
};
