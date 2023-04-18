#include <iostream>
#include <math.h>
#include <mpi.h>

using namespace std;

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
		for (int i = 0; i < threads_num; i++) {
			local_sum += arr[i];
		}
		cout << "Serial Sum is: " << local_sum << endl;
	}

	int recv_arr;
	MPI_Scatter(arr, 1, MPI_INT, &recv_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int local_sum = recv_arr;

	for (int h = 0; h < log2(threads_num); h++) {
		int tree_step = pow(2, h + 1);
		int child_step = pow(2, h);
		for (int send_id = child_step; send_id < threads_num; send_id += tree_step) {
			int recv_id = send_id - child_step;
			if (taskid == send_id) {
				MPI_Send(&local_sum, 1, MPI_INT, recv_id, 0, MPI_COMM_WORLD);
			}

			if (taskid == recv_id) {
				int recv_sum;
				MPI_Status status;
				MPI_Recv(&recv_sum, 1, MPI_INT, send_id, 0, MPI_COMM_WORLD, &status);
				local_sum += recv_sum;
			}
		}
	}
	
	// ¹ã²¥
	MPI_Bcast(&local_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
	cout << "Process " << taskid << " Sum is: " << local_sum << endl;
	
	delete[] arr;

	MPI_Finalize();
	return 0;
}