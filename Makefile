# Definir el compilador
NVCC = nvcc
CC = g++

# Definir las opciones de compilación
CFLAGS = -Xcompiler -fopenmp

# Nombre del archivo ejecutable
TARGET = prog

# Nombre del archivo fuente
SRC = prog.cu

# Compilador del programa
$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) $(CFLAGS)

# Limpiar los archivos generados
clean:
	rm -f $(TARGET)
