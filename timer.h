#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h> //test
#include <iostream>

class Timer
{
public:
    Timer()
    {
        Start();
    }

    double Timing(const std::string message = "", bool print = false)
    {
        gettimeofday(&end, nullptr);
        double timeuse = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec -
                          start.tv_usec) * 1.0 / 1000;

        if ("" != message)
        {
            if (print)
            {
                std::cout << "use time(" << message << "): "
                          << timeuse << "ms" << std::endl;
            }
        }
        Start();

        return timeuse;
    }

    std::string TimingStr(const std::string message = "", bool print = false)
    {
        gettimeofday(&end, nullptr);
        double timeuse = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec -
                          start.tv_usec) * 1.0 / 1000;

        std::ostringstream useTime("");

        useTime << "use time(" << message << "): "
                << timeuse << "ms";
        std::string useTimeStr = useTime.str();

        if (print)
        {
            std::cout << useTimeStr << std::endl;
        }

        Start();

        return useTimeStr;
    }


private:
    void Start()
    {
        gettimeofday(&start, nullptr);
    }

private:
    struct timeval start, end;
};


#endif // TIMER_H
