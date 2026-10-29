OMEGA ?= 3.33
general:
	g++ -std=c++17 -Wall -Wextra -O3 -mavx2 -maes -Wno-strict-aliasing -DOMEGA=$(OMEGA) general_leakage.cpp -o general

hamming:
	g++ -std=c++17 -Wall -Wextra -O3 -mavx2 -maes -Wno-strict-aliasing -fopenmp hamming_weight_leakage.cpp -o hamming -lm
	
clean:
	rm *.o main
