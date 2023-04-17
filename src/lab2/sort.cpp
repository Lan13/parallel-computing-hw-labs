#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int SIZE = 1 << 16;
int THREADS_NUM;
const int SAMPLE_SIZE = 8000;

//int temp[SIZE] = { 0 };

random_device rd;
mt19937 gen(rd());

void getArray(int *arr, int *arr1);
bool check(int arr[]);
void merge(int *arr, int left, int mid, int right);
void mergeSort(int *arr, int left, int right);

int main(int argc, char* argv[]) {

	// �α�ʾ��
	/*int arr[] = {15,46,48,93,39,6,72,91,14,
		36,69,40,89,61,97,12,21,54,
		53,97,84,58,32,27,33,72,20 };
	int arr1[] = { 15,46,48,93,39,6,72,91,14,
		36,69,40,89,61,97,12,21,54,
		53,97,84,58,32,27,33,72,20 };*/
	
	int taskid;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &THREADS_NUM);

	int* arr1 = NULL;
	if (taskid == 0) {
		int *arr = new int[SIZE];
		arr1 = new int[SIZE];
		getArray(arr, arr1);

		cout << "Array Size is: " << SIZE << endl;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		mergeSort(arr, 0, SIZE - 1);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Serial Correctness: " << check(arr) << endl;
		cout << "Serial Running Time: " << static_cast <float> (duration) / 1000000.0f << " seconds" << endl;

		delete[] arr;
	}

	int* task_arr = new int[SIZE / THREADS_NUM], task_sample[SAMPLE_SIZE];
	int task_stride = SIZE / THREADS_NUM;
	int sample_stride = SIZE / (SAMPLE_SIZE * THREADS_NUM);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// ���Ȼ���
	MPI_Scatter(arr1, task_stride, MPI_INT, task_arr, task_stride, MPI_INT, 0, MPI_COMM_WORLD);
	
	// �ֲ�����
	mergeSort(task_arr, 0, task_stride - 1);
	
	// �������
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		int arr_index = i * sample_stride;
		task_sample[i] = task_arr[arr_index];
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// ����������ѡ����Ԫ
	int *samples = NULL;
	int *pivots = new int[THREADS_NUM - 1];

	if (taskid == 0) {
		samples = new int[SAMPLE_SIZE * THREADS_NUM];
	}
	MPI_Gather(task_sample, SAMPLE_SIZE, MPI_INT, samples, SAMPLE_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	if (taskid == 0) {
		// ��������
		mergeSort(samples, 0, SAMPLE_SIZE * THREADS_NUM - 1);

		// ѡ����Ԫ
		for (int i = 0; i < THREADS_NUM - 1; i++) {
			int pivot_index = (i + 1) * SAMPLE_SIZE;
			pivots[i] = samples[pivot_index];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);	// �ȴ�0�Ž���ѡ����Ԫ

	// ѡ����Ԫ, �㲥
	MPI_Bcast(pivots, THREADS_NUM - 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	// ��Ԫ����
	int *part_start_index = new int[THREADS_NUM];
	int *part_len = new int[THREADS_NUM];
	int index = 0;
	for (int i = 0; i < THREADS_NUM - 1; i++) {
		part_start_index[i] = index;
		part_len[i] = 0;

		while ((index < task_stride) && (task_arr[index] <= pivots[i])) {
			index++;
			part_len[i]++;
		}
	}
	part_start_index[THREADS_NUM - 1] = index;
	part_len[THREADS_NUM - 1] = task_stride - index;

	// ȫ�ֽ���
	int *recv_part_len = new int[THREADS_NUM];
	MPI_Alltoall(part_len, 1, MPI_INT, recv_part_len, 1, MPI_INT, MPI_COMM_WORLD);

	int *recv_start_index = new int[THREADS_NUM];
	int recv_len = 0;
	for (int i = 0; i < THREADS_NUM; i++) {
		recv_start_index[i] = recv_len;
		recv_len += recv_part_len[i];
	}

	// ȫ�ֽ���
	int *recv_task_arr = new int[recv_len];
	MPI_Alltoallv(task_arr, part_len, part_start_index, MPI_INT, recv_task_arr, recv_part_len, recv_start_index, MPI_INT, MPI_COMM_WORLD);
	
	// �鲢����
	mergeSort(recv_task_arr, 0, recv_len - 1);
	int* sorted_arr = NULL;
	if (taskid == 0) {
		sorted_arr = new int[SIZE];
	}
	int *lens = new int[THREADS_NUM];
	MPI_Alltoall(&recv_len, 1, MPI_INT, lens, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Bcast(lens, THREADS_NUM, MPI_INT, 0, MPI_COMM_WORLD);

	int *len_index = new int[THREADS_NUM];
	int len = 0;
	for (int i = 0; i < THREADS_NUM; i++) {
		len_index[i] = len;
		len += lens[i];
	}

	MPI_Gatherv(recv_task_arr, recv_len, MPI_INT, sorted_arr, lens, len_index, MPI_INT, 0, MPI_COMM_WORLD);

	if (taskid == 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Parallel Correctness: " << check(sorted_arr) << endl;
		cout << "Parallel Running Time: " << static_cast <float> (duration) / 1000000.0f << " seconds" << endl;
	}


	delete[] arr1;
	delete[] task_arr;
	delete[] samples;
	delete[] pivots;
	delete[] part_start_index;
	delete[] part_len;
	delete[] recv_part_len;
	delete[] recv_start_index;
	delete[] recv_task_arr;
	delete[] sorted_arr;
	delete[] lens;
	delete[] len_index;

	MPI_Finalize();
	return 0;
}

void getArray(int* arr, int* arr1) {
	std::uniform_int_distribution<> distribute(0, SIZE);
	for (int i = 0; i < SIZE; i++) {
		int num = distribute(gen);
		arr[i] = num;
		arr1[i] = num;
	}
	return;
}

bool check(int arr[]) {
	for (int i = 0; i < SIZE - 1; i++) {
		if (arr[i] > arr[i + 1]) {
			return false;
		}
	}
	return true;
}

/*void merge(int* arr, int left, int mid, int right) {
	int l = left, r = mid + 1, index = left;
	while (l <= mid && r <= right) {
		if (arr[l] <= arr[r]) {
			temp[index++] = arr[l++];
		}
		else {
			temp[index++] = arr[r++];
		}
	}
	while (l <= mid) {
		temp[index++] = arr[l++];
	}
	while (r <= right) {
		temp[index++] = arr[r++];
	}
	for (int i = left; i <= right; i++) {
		arr[i] = temp[i];
	}
	return;
}*/

void mergeSort(int *arr, int left, int right) {
	if (left < right) {
		int mid = (right - left) / 2 + left;
		mergeSort(arr, left, mid);
		mergeSort(arr, mid + 1, right);
		//merge(arr, left, mid, right);
		std::inplace_merge(arr + left, arr + mid + 1, arr + right + 1);
	}
	return;
}