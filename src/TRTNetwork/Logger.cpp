#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "Logger.hpp"

namespace trt {

std::mutex LogTransaction::singletonMtx;

std::mutex Logger::singletonMtx;
std::unique_ptr<Logger> Logger::singletonLogger;

bool Logger::verbose = false;

/**
 * @note Generate timestamp string in the C way.
 */
std::string LogTransaction::genTimestamp()
{
    char szTime[32];
    time_t t = time(NULL);
    struct tm *ptm = localtime(&t);
    strftime(szTime, sizeof(szTime), "[%m/%d|%H:%M:%S] ", ptm);
    return std::string(szTime);
}

} // namespace trt
