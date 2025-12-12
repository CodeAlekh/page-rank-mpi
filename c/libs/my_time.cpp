#include <mpi.h>
#include <cstdio>

class Time {

    double start = 0.0;
    double time = 0.0;
    bool started = false;
    
    public:

    void start_time() {
        started = true;
        start = MPI_Wtime();
    }

    double get_time() {
        if (!started) {
            printf("Timer hasn't started. Please call start_time() before get_time()\n");
            throw("Timer hasn't started. Please call start_time() before get_time()");
        }
        time = MPI_Wtime() - start;
        return time;
    }
};