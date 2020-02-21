#pragma once

/**
 * This file implements a simple logging system to avoid complex
 * dependencies when using modern logging system, for example,
 * Google's logging system Glog.
 */

#include <string>
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>

#include "TensorRT/NvInfer.h"
#include "TensorRT/NvCaffeParser.h"

#define TRTLog(tag) trt::LogTransaction(tag).stream()

namespace trt {

enum Level
{
    INFO,
    WARN,
    ERROR
};

/**
 * @brief LogTransaction is used to process log.
 *
 *        Consider there are multiple threads that log simultaneosly.
 *        We implement this class to atomicly process the log.
 */
class LogTransaction
{
protected:
    /**
     * @brief A shared mutex for every transaction.
     */
    static std::mutex singletonMtx;

    /**
     * @note Generate timestamp string in the C way.
     * @todo Consider using a C++ approach in the future.
     */
    static std::string genTimestamp();

    /**
     * @brief Each transaction will acquire the ownership of singletonMtx.
     *        This is to ensure only one trasaction is active at a time.
     */
    std::unique_lock<std::mutex> lock;

public:
    LogTransaction(const std::string& tag)
        : lock(singletonMtx)
    {
        std::cout << tag << genTimestamp();
    }

    LogTransaction(int level)
        : lock(singletonMtx)
    {
        if (level == INFO)
            std::cout << "[I]" << genTimestamp();
        else if (level == WARN)
            std::cout << "[W]" << genTimestamp();
        else
            std::cout << "[E]" << genTimestamp();
    }

    ~LogTransaction()
    {
        std::cout << std::endl;
    }

    std::ostream& stream()
    {
        return std::cout;
    }
};

/**
 * @brief Logger for the neural network instance.
 *
 *        TensorRT exposes a logger interface to pass a user-provided
 *        logger to show the status of the execution in TensorRT.
 *        For resource saving purpose, we use a global singleton logger
 *        to receive and process the logs.
 */
class Logger : public nvinfer1::ILogger
{
protected:
    static std::mutex singletonMtx;
    static std::unique_ptr<Logger> singletonLogger;
    static bool verbose;

public:
    static Logger& globalInstance()
    {
        std::lock_guard<std::mutex> locker(singletonMtx);
        if (!singletonLogger)
            singletonLogger.reset(new Logger());
        return *singletonLogger.get();
    }

    Logger()
    {
        if (verbose)
            TRTLog(INFO) << "Created TensorRT Logger";
    }

    void log(Severity severity, const char *msg)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            TRTLog(ERROR) << msg;
            return;
        case Severity::kWARNING:
            TRTLog(WARN) << msg;
            return;
        case Severity::kINFO:
            if (verbose)
                TRTLog(INFO) << msg;
            return;
        }
    }
};

} // namespace trt
