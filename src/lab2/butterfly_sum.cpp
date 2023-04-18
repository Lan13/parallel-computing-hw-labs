#include <iostream>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
	int taskid, threads_num;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &threads_num);

	int* arr = NULL;

	if (taskid == 0) {
		arr = new int[threads_num];
		for (int i = 0; i < threads_num; i++) {
			arr[i] = i + 1;
		}

		int local_sum = 0;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		for (int i = 0; i < threads_num; i++) {
			local_sum += arr[i];
		}
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Serial Sum is: " << local_sum << endl;
		cout << "Serial Running Time: " << static_cast <float> (duration) / 1000000.0f << " seconds" << endl;
	}

	int recv_arr;
	MPI_Scatter(arr, 1, MPI_INT, &recv_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int local_sum = recv_arr;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int h = 0; h < ceil(log2(threads_num)); h++) {
		int butterfly_step = pow(2, h + 1);
		int pair_step = pow(2, h);
		for (int start_id = 0; start_id < threads_num; start_id += butterfly_step) {
			for (int id = start_id; id < start_id + pair_step; id++) {
				int pair_id = id + pair_step;

				if (pair_id >= threads_num) {
					break;
				}
				
				if (taskid == id) {
					int recv_sum;
					MPI_Status status;
					MPI_Send(&local_sum, 1, MPI_INT, pair_id, 0, MPI_COMM_WORLD);
					MPI_Recv(&recv_sum, 1, MPI_INT, pair_id, 0, MPI_COMM_WORLD, &status);
					local_sum += recv_sum;
				}

				if (taskid == pair_id) {
					int recv_sum;
					MPI_Status status;
					MPI_Send(&local_sum, 1, MPI_INT, id, 0, MPI_COMM_WORLD);
					MPI_Recv(&recv_sum, 1, MPI_INT, id, 0, MPI_COMM_WORLD, &status);
					local_sum += recv_sum;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	cout << "Process " << taskid << " Sum is: " << local_sum << endl;
	if (taskid == 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Serial Running Time: " << static_cast <float> (duration) / 1000000.0f << " seconds" << endl;
	}

	delete[] arr;
	MPI_Finalize();
	return 0;
}