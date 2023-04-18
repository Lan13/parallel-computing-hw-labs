#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

const int SIZE = 1 << 24;
const int NUM_THREADS = 16;
const int SAMPLE_SIZE = 8000;

random_device rd;
mt19937 gen(rd());

vector<int> arr(SIZE), arr1(SIZE);
vector<int> temp(SIZE);

void getArray(vector<int>& arr, vector<int>& arr1);
bool check(vector<int> arr);
void merge(vector<int>& arr, int left, int mid, int right);
void mergeSort(vector<int>& arr, int left, int right);
void regularSample(vector<int> arr, vector<int>& samples, vector<int>& pivots);
vector<int> pivotPartition(vector<int> arr, vector<int> pivots, vector<int>& accumulate_counts);
void mergeSortParallel(vector<int>& arr);


int main() {
	// 课本示例
	/*vector<int> arr = {15,46,48,93,39,6,72,91,14,
		36,69,40,89,61,97,12,21,54,
		53,97,84,58,32,27,33,72,20 };
	vector<int> arr1 = { 15,46,48,93,39,6,72,91,14,
		36,69,40,89,61,97,12,21,54,
		53,97,84,58,32,27,33,72,20 };*/
	
	getArray(arr, arr1);

	double start = omp_get_wtime();
	mergeSort(arr, 0, SIZE - 1);
	double end = omp_get_wtime();
	cout << "Correctness: " << check(arr) << endl;
	cout << "Serial Running Time: " << end - start << " seconds" << endl;

	start = omp_get_wtime();
	mergeSortParallel(arr1);
	end = omp_get_wtime();
	cout << "Correctness: " << check(arr1) << endl;
	cout << "Parallel Running Time: " << end - start << " seconds" << endl;
	return 0;
}

void getArray(vector<int>& arr, vector<int>& arr1) {
	std::uniform_int_distribution<> distribute(0, SIZE);
	for (int i = 0; i < SIZE; i++) {
		int num = distribute(gen);
		arr[i] = num;
		arr1[i] = num;
	}
	return;
}

bool check(vector<int> arr) {
	for (int i = 0; i < SIZE - 1; i++) {
		if (arr[i] > arr[i + 1]) {
			return false;
		}
	}
	return true;
}

void merge(vector<int>& arr, int left, int mid, int right) {
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
}

void mergeSort(vector<int>& arr, int left, int right) {
	if (left < right) {
		int mid = (right - left) / 2 + left;
		mergeSort(arr, left, mid);
		mergeSort(arr, mid + 1, right);
		merge(arr, left, mid, right);
	}
	return;
}

void regularSample(vector<int> arr, vector<int>& samples, vector<int>& pivots) {
	int stride = SIZE / (SAMPLE_SIZE * NUM_THREADS);

	// 正则采样
	#pragma omp parallel num_threads(NUM_THREADS) shared(arr, samples) 
	{
		int tid = omp_get_thread_num();
		// 记录不同处理器采样后放入的起始位置
		int thread_sample_index = tid * SAMPLE_SIZE;
		// 记录不同处理器采样前从arr中取出数据的位置
		int thread_arr_index = tid * SAMPLE_SIZE * stride;
		for (int i = 0; i < SAMPLE_SIZE; i++) {
			int sample_index = i + thread_sample_index;
			int arr_index = i * stride + thread_arr_index;
			samples[sample_index] = arr[arr_index];
		}
	#pragma omp barrier
	}

	// 采样排序
	mergeSort(samples, 0, SAMPLE_SIZE * NUM_THREADS - 1);

	// 选择主元
	for (int i = 0; i < NUM_THREADS - 1; i++) {
		int pivot_index = (i + 1) * SAMPLE_SIZE;
		pivots[i] = samples[pivot_index];
	}
	return;
}


/*
* arr_change用来存储全局交换后的数组
* lens记录每个处理器中划分之后的每段长度
* counts记录全局交换后的每个处理器中应该处理的数组长度
* accumulate_counts是counts的累计和，方便将arr中的数据交换到temp中，用作偏移地址
*/
vector<int> pivotPartition(vector<int> arr, vector<int> pivots, vector<int>& accumulate_counts) {
	// 主元划分
	vector<int> arr_change(SIZE);
	vector<vector<int> > lens(NUM_THREADS, vector<int>(NUM_THREADS));
	vector<int> counts(NUM_THREADS);

	#pragma omp parallel num_threads(NUM_THREADS) shared(arr, temp, lens, counts, accumulate_counts)
	{
		int tid = omp_get_thread_num();
		int stride = SIZE / NUM_THREADS;
		int l = tid * stride, r = (tid + 1) * stride - 1;
		vector<int> partitions(NUM_THREADS + 1);
		partitions[0] = l - 1;
		for (int i = 1; i < NUM_THREADS + 1; i++)
			partitions[i] = r;
		//partitions[NUM_THREADS] = r;


		// 主元划分
		int ll = l;
		for (int i = 0; i < NUM_THREADS - 1; i++) {
			for (int j = ll; j <= r; j++) {
				if (arr[j] > pivots[i]) {
					partitions[i + 1] = j - 1;
					ll = j;
					break;
				}
			}
		}

		// 这一段程序的目的是为了计算交换后的地址索引，注意这一块需要同步路障
		lens[tid][NUM_THREADS - 1] = r - ll + 1;
		for (int i = 0; i < NUM_THREADS; i++) {
			lens[tid][i] = partitions[i + 1] - partitions[i];
		}
		#pragma omp barrier

		for (int i = 0; i < NUM_THREADS; i++) {
			counts[tid] += lens[i][tid];
		}
		#pragma omp barrier

		for (int i = 0; i <= tid; i++) {
			accumulate_counts[tid] += counts[i];
		}
		#pragma omp barrier
		// 这一段程序的目的是为了计算交换后的地址索引

		// 全局交换
		for (int i = 0; i < NUM_THREADS; i++) {
			int dest_index = (i == 0) ? 0 : accumulate_counts[i - 1];
			for (int ii = 0; ii < tid; ii++) {
				dest_index += lens[ii][i];
			}
			for (int j = partitions[i] + 1, k = 0; j <= partitions[i + 1]; j++, k++) {
				arr_change[dest_index + k] = arr[j];
			}
		}
	}
	return arr_change;
}

void mergeSortParallel(vector<int>& arr) {
	vector<int> samples(SAMPLE_SIZE * NUM_THREADS);
	vector<int> pivots(NUM_THREADS - 1);
	vector<int> accumulate_counts(NUM_THREADS);
	omp_set_num_threads(NUM_THREADS);

	// 均匀划分且局部排序
	#pragma omp parallel num_threads(NUM_THREADS) shared(arr)
	{
		int tid = omp_get_thread_num();
		int stride = SIZE / NUM_THREADS;
		int l = tid * stride, r = (tid + 1) * stride - 1;
		mergeSort(arr, l, r);
	}
	#pragma omp barrier

	// 正则采样且采样排序且选择主元
	regularSample(arr, samples, pivots);
	#pragma omp barrier

	// 主元划分且全局交换
	arr = pivotPartition(arr, pivots, accumulate_counts);
	#pragma omp barrier

	// 归并排序
	#pragma omp parallel num_threads(NUM_THREADS) shared(arr)
	{
		int tid = omp_get_thread_num();
		int stride = SIZE / NUM_THREADS;
		int l = (tid == 0) ? 0 : accumulate_counts[tid - 1], r = accumulate_counts[tid] - 1;
		mergeSort(arr, l, r);
	}

	return;
}