#ifndef ILOGGER_HPP
#define ILOGGER_HPP

#define INFO_FATAL(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_FATAL, __VA_ARGS__)
#define INFO_ERROR(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_ERROR, __VA_ARGS__)
#define INFO_WARNING(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_WARNING, __VA_ARGS__)
#define INFO(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_INFO, __VA_ARGS__)
#define INFO_VERBOSE(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_VERBOSE, __VA_ARGS__)
#define INFO_DEBUG(...) Logger::log_(__FILE__, __LINE__, Logger::LogLevel::LOG_DEBUG, __VA_ARGS__)

namespace Logger
{
    enum class LogLevel: int
    {
        LOG_FATAL,  // 0
        LOG_ERROR,
        LOG_WARNING,
        LOG_INFO,
        LOG_VERBOSE,
        LOG_DEBUG
    };

    void setLogLevel(LogLevel level);

    LogLevel getLogLevel();

    void log_(const char *file_name, int line, LogLevel level, const char *format, ...);
};


#endif  // ILOGGER_HPP