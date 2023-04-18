#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

const int NUM_STEPS = 100000;
const int NUM_THREADS = 8;

double serialPi();
double parallelPi1();
double parallelPi2();
double parallelPi3();
double parallelPi4();

int main() {
	double start = omp_get_wtime();
	double pi = serialPi();
	double end = omp_get_wtime();
	cout << "Serial Pi: " << pi << endl;
	cout << "Serial Running Time: " << end - start << " seconds" << endl << endl;

	start = omp_get_wtime();
	double pi1 = parallelPi1();
	end = omp_get_wtime();
	cout << "Parallel Pi: " << pi1 << endl;
	cout << "Parallel Running Time: " << end - start << " seconds" << endl << endl;

	start = omp_get_wtime();
	double pi2 = parallelPi2();
	end = omp_get_wtime();
	cout << "Parallel Pi: " << pi2 << endl;
	cout << "Parallel Running Time: " << end - start << " seconds" << endl << endl;

	start = omp_get_wtime();
	double pi3 = parallelPi3();
	end = omp_get_wtime();
	cout << "Parallel Pi: " << pi3 << endl;
	cout << "Parallel Running Time: " << end - start << " seconds" << endl << endl;

	start = omp_get_wtime();
	double pi4 = parallelPi4();
	end = omp_get_wtime();
	cout << "Parallel Pi: " << pi4 << endl;
	cout << "Parallel Running Time: " << end - start << " seconds" << endl << endl;
	return 0;
}

double serialPi() {
	double x, pi, sum = 0.0;
	double step = 1.0 / NUM_STEPS;
	for (int i = 0; i < NUM_STEPS; i++) {
		x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	return pi;
}

double parallelPi1() {
	double pi = 0.0, sum[NUM_THREADS] = { 0.0 };
	double step = 1.0 / NUM_STEPS;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel num_threads(NUM_THREADS) shared(sum)
	{
		double x;
		int tid = omp_get_thread_num();
		for (int i = 0; i < NUM_STEPS; i += NUM_THREADS) {
			x = (i + 0.5) * step;
			sum[tid] += 4.0 / (1.0 + x * x);
		}
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		pi += step * sum[i];
	}
	return pi;
}

double parallelPi2() {
	double pi = 0.0, sum[NUM_THREADS] = { 0.0 };
	double step = 1.0 / NUM_STEPS;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel num_threads(NUM_THREADS) shared(sum)
	{
		double x;
		int tid = omp_get_thread_num();
		#pragma omp for
		for (int i = 0; i < NUM_STEPS; i++) {
			x = (i + 0.5) * step;
			sum[tid] += 4.0 / (1.0 + x * x);
		}
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		pi += step * sum[i];
	}
	return pi;
}

double parallelPi3() {
	double pi = 0.0, x = 0.0, sum = 0.0;
	double step = 1.0 / NUM_STEPS;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel num_threads(NUM_THREADS) private(x, sum)
	{
		int tid = omp_get_thread_num();
		sum = 0.0;
		for (int i = tid; i < NUM_STEPS; i += NUM_THREADS) {
			x = (i + 0.5) * step;
			sum += 4.0 / (1.0 + x * x);
		}
		#pragma omp critical
			pi += sum * step;
	}
	return pi;
}

double parallelPi4() {
	double pi = 0.0, x = 0.0, sum = 0.0;
	double step = 1.0 / NUM_STEPS;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS) private(x)
	for (int i = 0; i < NUM_STEPS; i++) {
		x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	return pi;
}