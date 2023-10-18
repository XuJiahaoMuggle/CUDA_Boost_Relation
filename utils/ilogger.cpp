#include "ilogger.hpp"

#include <ctime>
#include <cstdarg>
#include <cstdio>

#ifdef WIN32
#include <Windows.h>
#endif

#include <string>
using namespace std;

namespace Logger
{
    static LogLevel global_level_ = LogLevel::LOG_INFO;
    
    const char *levelString(LogLevel level)
    {
        switch (level)
        {
        case LogLevel::LOG_FATAL:
            return "LOG_FATAL";
            break;
        case LogLevel::LOG_ERROR:
            return "LOG_ERROR";
            break;
        case LogLevel::LOG_WARNING:
            return "LOG_WARNING";
            break;
        case LogLevel::LOG_INFO:
            return "LOG_INFO";
            break;
        case LogLevel::LOG_VERBOSE:
            return "LOG_VERBOSE";
            break;
        case LogLevel::LOG_DEBUG:
            return "LOG_DEBUG";
            break;
        default:
            return "UNKNOWN";
            break;
        }
    }

    void setLogLevel(LogLevel level)
    {
        global_level_ = level;
    }

    LogLevel getLogLevel()
    {
        return global_level_;
    }
    
    static const string getTime()
    {
        char time_string[20];
        time_t timep;
        time(&timep);
        tm &t = *(tm *)localtime(&timep);
        sprintf(
            time_string,
            "%04d-%02d-%02d %02d:%02d:%02d", 
            t.tm_year + 1900, 
            t.tm_mon + 1, 
            t.tm_mday, 
            t.tm_hour, 
            t.tm_min, 
            t.tm_sec
        );
        return time_string;
    }
    
    #ifdef WIN32
    static void setTextColor(WORD attribute)
    {
        HANDLE consolehwnd = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(consolehwnd, attribute);
    }
    // TODO: for unix system.
    #else
    static void setTextColor(){}
    #endif

    void log_(const char *file_name, int line, LogLevel level, const char *format, ...)
    {
        if (level > global_level_)
            return ;

        va_list vl;
        va_start(vl, format);
        char buffer[2048];
        const string cur_time = Logger::getTime();  

        // save copy.
        int n = snprintf(buffer, sizeof(buffer), "[%s]", cur_time.c_str());
        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", levelString(level));
        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]", file_name, line);
        vsnprintf(buffer + n, sizeof(buffer) - n, format, vl);
        switch (level)
        {
        case LogLevel::LOG_FATAL:
        case LogLevel::LOG_ERROR:
            setTextColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
            break;
        case LogLevel::LOG_WARNING:
            setTextColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            break;
        case LogLevel::LOG_INFO:
        case LogLevel::LOG_VERBOSE:
        case LogLevel::LOG_DEBUG:
            setTextColor(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            break;
        default:
            setTextColor(0X0007);
            break;
        }
        fprintf(stdout, "%s\n", buffer);
        va_end(vl);
        if (level == LogLevel::LOG_FATAL)
        {
            fflush(stdout);
            setTextColor(0X0007);
            abort();
        }
        setTextColor(0X0007);
    }
};