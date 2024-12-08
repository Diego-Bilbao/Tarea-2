#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Funcion para imprimir el array, ecibe un array arr[] y su tamaño size. Recorre el array e imprime cada elemento, separado por espacios,
// seguido de un salto de línea al final
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Funcion para generar un array aleatorio, utilizando time como semilla generadora de numeros aleatorios y rand() que genera numeros aleatorios
// entre 0 y 999
void generateRandomArray(int* arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000;
    }
}

// FUncion auxiliar de Quicksor, toma un array, un indice bajo y un indice alto, selecciona un pivote y reorganiza los elementos del array de manera
// que los elementos menores al pivote queden a la izquierda y los mayores a la derecha, por ultimo devuelve el indice de la nueva posicion del pivote
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Quick Sort paralelo en CPU, si el rango de indices es valido, divide el array recursivamente y ordena las dos mitades de forma paralela usando
// la directiva #pragma omp parallel sections
void quickSortParallel(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSortParallel(arr, low, pi - 1);
            }
            #pragma omp section
            {
                quickSortParallel(arr, pi + 1, high);
            }
        }
    }
}

// Kernel para realizar la fusion de dos segmentos de un array, este kernel se ejecuta en paralelo en la GPU, donde cada hilo fusiona dos segmentos
// del array, luego se usa un paso (step) para controlar el tamaño de los bloques a fusionar, la cual se realiza comparando elementos entre los
// dos bloques y colocando los mas pequeños en un array temporal
__global__ void mergeSortKernel(int* d_data, int* d_temp, int size, int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = idx * step * 2;
    if (start < size) {
        int mid = start + step;
        int end = start + 2 * step;

        // Si el final del array esta fuera de rango
        if (mid > size) mid = size;
        if (end > size) end = size;

        // Realizamos la fusion de dos partes
        int i = start;
        int j = mid;
        int k = start;

        while (i < mid && j < end) {
            if (d_data[i] <= d_data[j]) {
                d_temp[k] = d_data[i];
                i++;
            } else {
                d_temp[k] = d_data[j];
                j++;
            }
            k++;
        }

        // Copiar el resto de los elementos
        while (i < mid) {
            d_temp[k] = d_data[i];
            i++;
            k++;
        }

        while (j < end) {
            d_temp[k] = d_data[j];
            j++;
            k++;
        }

        // Copiar el resultado de vuelta a d_data
        for (int i = start; i < end; i++) {
            d_data[i] = d_temp[i];
        }
    }
}

// Merge Sort en GPU, el array se transfiere desde la memoria del host a la memoria del la GPU, para luego ejecutar el kernel de fusion en bloques
// de hilos paralelos, donde a medida que el algoritmo avanza, el tamaño de los segmentos a fusionar crece exponencialmente, asi, al terminar, el
// array ordenado se transfiere de vuelta al host al finalizar
void mergeSort(int* h_data, int size) {
    int* d_data, *d_temp;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_temp, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    for (int step = 1; step < size; step *= 2) {
        mergeSortKernel<<<blocks, threadsPerBlock>>>(d_data, d_temp, size, step);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_temp);
}

// El programa recibe tres parametros, el tamaño del array n, el modo de ejecucion, y el numero de hilos en el caso de la CPU, dependiendo del
// modo, se ejecuta Quick Sort paralelo (CPU) o Merge Sort (GPU), luego se generan datos aleatorios, se ordenan y se imprime el tiempo de ejecucion
// de cada algoritmo.
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Uso: ./prog <n> <modo> <nt>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);  // Tamaño del array
    int mode = std::atoi(argv[2]); // Modo: 0 para CPU, 1 para GPU
    int nt = std::atoi(argv[3]);   // Numero de hilos para OpenMP (modo CPU)

    int* h_data = new int[n];
    int* h_data_copy = new int[n];

    generateRandomArray(h_data, n);
    std::copy(h_data, h_data + n, h_data_copy);

    std::cout << "Array generado: ";
    printArray(h_data, n);

    if (mode == 0) {  // CPU
        // Ordenar usando Quick Sort Paralelo, se configura el numero de hilos para OpenMP
        omp_set_num_threads(nt);
        double start = omp_get_wtime();
        quickSortParallel(h_data, 0, n - 1);
        double end = omp_get_wtime();
        std::cout << "Array ordenado con Quick Sort Paralelo (CPU): ";
        printArray(h_data, n);
        std::cout << "Tiempo de ejecución Quick Sort en CPU: " << (end - start) << " segundos\n";
    } 
    else if (mode == 1) {  // GPU
        // Ordenar usando Merge Sort en GPU
        double start = omp_get_wtime();
        mergeSort(h_data_copy, n);
        double end = omp_get_wtime();
        std::cout << "Array ordenado con Merge Sort Paralelo (GPU): ";
        printArray(h_data_copy, n);
        std::cout << "Tiempo de ejecución Merge Sort en GPU: " << (end - start) << " segundos\n";
    }
    else {
        std::cout << "Modo inválido. Use 0 para CPU o 1 para GPU.\n";
    }

    delete[] h_data;
    delete[] h_data_copy;
    return 0;
}

