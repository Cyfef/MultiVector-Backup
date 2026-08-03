#ifndef TIMECOUNTER_H
#define TIMECOUNTER_H

#include <chrono>
#include <cassert>

class TimeCounter {
public:
    TimeCounter() {
        isSet = false;
    }
    void setTime() {
        pigeon_local_time_stamp = std::chrono::high_resolution_clock::now();
        isSet = true;
    }

    int catchTime() {
        auto curStamp = std::chrono::high_resolution_clock::now();
        assert(isSet);
        isSet = false;
        return std::chrono::duration_cast<std::chrono::milliseconds>(curStamp - pigeon_local_time_stamp).count();
    }

private:
    std::chrono::high_resolution_clock::time_point pigeon_local_time_stamp;
    bool isSet;
};

#endif