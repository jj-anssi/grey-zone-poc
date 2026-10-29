#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <omp.h> 
#include <boost/math/special_functions/binomial.hpp>

#include <x86intrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

// Information in bits contained in a HW of a 32-bit word
#define OMEGA 3.33

using namespace std;

typedef struct
{
    uint32_t A;
    uint32_t B;
    uint32_t C;
} triplet_t;

const uint8_t S[] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

const __m128i SR = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,15,10,5);
const __m128i INV_SR = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,10,13,0,7);
const __m128i ZERO = _mm_set_epi32(0, 0, 0, 0);

double
C_32_k(const int k)
{
    return boost::math::binomial_coefficient<double>(32, k);
}

void
keySchedule(uint32_t sk[44], double leaks[44])
{
    /* Draw a random master key */
    uint8_t k[16];
    int fd = open("/dev/urandom", O_RDONLY);

    if (read(fd, k, 16) <= 0) {
        perror("read");
        exit(1);
    }

    close(fd);

	for (int j = 0; j < 4; ++j) {
		sk[j] =
			  ((uint32_t)(k[4*j + 0]) << 24)
			^ ((uint32_t)(k[4*j + 1]) << 16)
			^ ((uint32_t)(k[4*j + 2]) <<  8)
			^ ((uint32_t)(k[4*j + 3]) <<  0)
		;
	}

	for (int j = 4; j < 44; ++j) {
		if ((j % 4) == 0) {
            /* We ignore round constants for simplicity */
			sk[j] = sk[j-4] ^
				  (((uint32_t)S[ ((sk[j-1]) >> 16) & 0xff ] ) << 24)
				^ (((uint32_t)S[ ((sk[j-1]) >>  8) & 0xff ] ) << 16)
				^ (((uint32_t)S[ ((sk[j-1]) >>  0) & 0xff ] ) <<  8)
				^ (((uint32_t)S[ ((sk[j-1]) >> 24) & 0xff ] ) <<  0)
			;
		} else {
			sk[j] = sk[j-4] ^ sk[j-1];
		}
	}

	for (int j = 0; j < 44; ++j) {
		leaks[j] = (double)__builtin_popcount(sk[j]);
	}
}

int
my_popcount(__m128i A)
{
	const uint32_t x = *(const uint32_t *)&A;
	return __builtin_popcount(x);
}

void
print128(const char *s, __m128i A)
{
    uint8_t *arr = (uint8_t *)&A;
	printf("%s ", s);
    for (int i= 0; i < 16; ++i) {
    	printf("%02x ", arr[i]);
    }
    printf("\n");
}

vector<triplet_t>
getRedCandidates(const uint32_t x4, const __m128i& X4, const __m128i& Y4, double leaks[44])
{
    (void) Y4;

    vector<triplet_t> candidates_red;

    __m128i X4_ = _mm_shuffle_epi8(X4, SR); // SR

    for (uint64_t x0 = 0; x0 < (1ul << 32); ++x0) {
        if (__builtin_popcount(x0) != leaks[0*4+3]) {
            continue; // X0
        }
        const __m128i X0 = _mm_set_epi32 (0, 0, 0, x0);
        const __m128i Y3 = _mm_xor_si128(X0, X4_);

        __m128i X3 = _mm_aesdeclast_si128(Y3, ZERO);
        __m128i T = _mm_shuffle_epi8(X3, SR);
        if (my_popcount(T) != leaks[3*4+3]) {
            continue; // X3
        }

        const uint32_t x3 = (uint32_t)_mm_extract_epi32(T, 0);

        if (__builtin_popcount(x3 ^ x4) != leaks[4*4+2]) {
            continue; // C4
        }

        triplet_t red = {
            .A = (uint32_t) x0,
            .B = (uint32_t) x3,
            .C = (uint32_t) x4,
        };
        candidates_red.push_back(red);
    }

    return candidates_red;
}

vector<triplet_t>
getGreenCandidates(const uint32_t x4, const __m128i& X4, const __m128i& Y4, double leaks[44])
{
    (void) Y4;

    vector<triplet_t> candidates_green;

    __m128i X4_ = _mm_shuffle_epi8(X4, SR); // SR

    for (uint64_t x8 = 0; x8 < (1ul << 32); ++x8) {
        if (__builtin_popcount(x8) != leaks[8*4+3]) {
            continue; // X8
        }
        const __m128i X8 = _mm_set_epi32 (0, 0, 0, x8);
        const __m128i Y7 = _mm_xor_si128(X8, X4_);

        __m128i X7 = _mm_aesdeclast_si128(Y7, ZERO);
        __m128i T = _mm_shuffle_epi8(X7, SR);
        if (my_popcount(T) != leaks[7*4+3]) {
            continue; // X7
        }

        const uint32_t x7 = (uint32_t)_mm_extract_epi32(T, 0);
        if (__builtin_popcount(x7 ^ x8) != leaks[8*4+2]) {
            continue; // C8
        }

        __m128i Y8 = _mm_aesenclast_si128(X8, ZERO);
        __m128i U = _mm_shuffle_epi8(Y8, _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,7,10,13,0));

        // Blue parity filter
        if (my_popcount(U) % 2 != int(leaks[5 * 4 + 3] + leaks[9 * 4 + 3]) % 2) { // leaks[5*4+3] already used
            continue;
        }

        triplet_t green = {
            .A = (uint32_t) x4,
            .B = (uint32_t) x7,
            .C = (uint32_t) x8,
        };
        candidates_green.push_back(green);
    }

    return candidates_green;
}

vector<triplet_t>
getBlackCandidates(const uint32_t x4, const __m128i& X4, const __m128i& Y4, double leaks[44])
{
    (void) X4;

    vector<triplet_t> candidates_black;

    for (uint64_t x5 = 0; x5 < (1ul << 32); ++x5) {
        if (__builtin_popcount(x5) != leaks[5*4+3]) {
            continue; // X5
        }
        const __m128i X5 = _mm_set_epi32 (0, 0, 0, x5);
        const __m128i X1 = _mm_xor_si128(X5, Y4);
        if (my_popcount(X1) != leaks[1*4+3]) {
            continue; // X1
        }

        if (__builtin_popcount(x4 ^ x5) != leaks[5*4+2]) {
            continue; // C5
        }

        const uint32_t x1 = (uint32_t)_mm_extract_epi32(X1, 0);

        triplet_t black = {
            .A = (uint32_t) x1,
            .B = (uint32_t) x4,
            .C = (uint32_t) x5,
        };
        candidates_black.push_back(black);
    }

    return candidates_black;
}

int
finalChecks(double leaks[44],
            const uint32_t x0, const uint32_t x1, const uint32_t x3,
            const uint32_t x4, const uint32_t t4, const uint32_t x5,
            const uint32_t x6, const uint32_t x7, const uint32_t x8,
            uint32_t& sk0_0, uint32_t& sk0_1, uint32_t& sk0_2, uint32_t& sk0_3
            )
{
    const uint32_t sk1_2 = x0 ^ x1;
    if (__builtin_popcount(sk1_2) != leaks[1*4+2]) {
        return 0; // C1
    }

    const uint32_t sk4_2 = x4 ^ x3;       // already filtered
    const uint32_t sk4_3 = x4;            // already filtered

    // Check all remaining 2w in K5
    const uint32_t sk5_3 = x5;            // already filtered
    const uint32_t sk5_2 = sk5_3 ^ sk4_3; // already filtered
    const uint32_t sk5_1 = sk5_2 ^ sk4_2;
    if (__builtin_popcount(sk5_1) != leaks[5*4+1]) {
        return 0;
    }

    // Check remaining 3w in K6
    const uint32_t sk6_3 = x6;            // already filtered
    const uint32_t sk6_2 = sk6_3 ^ sk5_3;
    if (__builtin_popcount(sk6_2) != leaks[6*4+2]) {
        return 0;
    }
    const uint32_t sk6_1 = sk6_2 ^ sk5_2;  // already filtered
    const uint32_t sk6_0 = sk6_1 ^ sk5_1;
    if (__builtin_popcount(sk6_0) != leaks[6*4+0]) {
        return 0;
    }

    // Check remaining 2w in K7
    const uint32_t sk7_3 = x7;            // already filtered
    const uint32_t sk7_2 = sk7_3 ^ sk6_3; // already filtered
    const uint32_t sk7_1 = sk7_2 ^ sk6_2;
    if (__builtin_popcount(sk7_1) != leaks[7*4+1]) {
        return 0;
    }
    const uint32_t sk7_0 = sk7_1 ^ sk6_1;
    if (__builtin_popcount(sk7_0) != leaks[7*4+0]) {
        return 0;
    }

    // Check remaining 8w in K8
    const uint32_t sk8_3 = x8;            // already filtered
    const uint32_t sk8_2 = sk8_3 ^ sk7_3;
    if (__builtin_popcount(sk8_2) != leaks[8*4+2]) {
        return 0;
    }
    const uint32_t sk8_1 = sk8_2 ^ sk7_2; // already filtered
    const uint32_t sk8_0 = sk8_1 ^ sk7_1;
    if (__builtin_popcount(sk8_0) != leaks[8*4+0]) {
        return 0;
    }

    // Complete checks on K5
    const __m128i X5 = _mm_set_epi32 (0, 0, 0, x5);
    __m128i Y5 = _mm_aesenclast_si128(X5, ZERO);
    __m128i T5 = _mm_shuffle_epi8(Y5, INV_SR);
    const uint32_t t5 = (uint32_t)_mm_extract_epi32(T5, 0);
    const uint32_t sk5_0 = sk6_0 ^ t5;
    if (__builtin_popcount(sk5_0) != leaks[5*4+0]) {
        return 0;
    }

    // Complete checks on K4
    const uint32_t sk4_1 = sk5_0 ^ sk5_1;
    if (__builtin_popcount(sk4_1) != leaks[4*4+1]) {
        return 0;
    }
    const uint32_t sk4_0 = sk5_0 ^ t4;
    if (__builtin_popcount(sk4_0) != leaks[4*4+0]) {
        return 0;
    }

    // Complete checks on K3
    const uint32_t sk3_3 = x3;
    const uint32_t sk3_2 = sk4_1 ^ sk4_2;
    if (__builtin_popcount(sk3_2) != leaks[3*4+2]) {
        return 0;
    }
    const uint32_t sk3_1 = sk4_0 ^ sk4_1;
    if (__builtin_popcount(sk3_1) != leaks[3*4+1]) {
        return 0;
    }

    const __m128i X3 = _mm_set_epi32 (0, 0, 0, x3); // already filtered
    __m128i Y3 = _mm_aesenclast_si128(X3, ZERO);
    __m128i T3 = _mm_shuffle_epi8(Y3, INV_SR);
    const uint32_t t3 = (uint32_t)_mm_extract_epi32(T3, 0);
    const uint32_t sk3_0 = sk4_0 ^ t3;
    if (__builtin_popcount(sk3_0) != leaks[3*4+0]) {
        return 0;
    }

    // Complete checks on K2
    const uint32_t sk2_3 = sk3_3 ^ sk3_2;
    if (__builtin_popcount(sk2_3) != leaks[2*4+3]) {
        return 0;
    }

    const uint32_t sk2_2 = sk3_1 ^ sk3_2;
    if (__builtin_popcount(sk2_2) != leaks[2*4+2]) {
        return 0;
    }

    const uint32_t sk2_1 = sk3_0 ^ sk3_1;
    if (__builtin_popcount(sk2_1) != leaks[2*4+1]) {
        return 0;
    }

    const __m128i X2 = _mm_set_epi32 (0, 0, 0, sk2_3);
    __m128i Y2 = _mm_aesenclast_si128(X2, ZERO);
    __m128i T2 = _mm_shuffle_epi8(Y2, INV_SR);
    const uint32_t t2 = (uint32_t)_mm_extract_epi32(T2, 0);

    const uint32_t sk2_0 = sk3_0 ^ t2;
    if (__builtin_popcount(sk2_0) != leaks[2*4+0]) {
        return 0;
    }

    // Complete checks on K1
    const uint32_t sk1_1 = sk2_0 ^ sk2_1;
    if (__builtin_popcount(sk1_1) != leaks[1*4+1]) {
        return 0;
    }

    const __m128i X1 = _mm_set_epi32 (0, 0, 0, x1); // already filtered
    __m128i Y1 = _mm_aesenclast_si128(X1, ZERO);
    __m128i T1 = _mm_shuffle_epi8(Y1, INV_SR);
    const uint32_t t1 = (uint32_t)_mm_extract_epi32(T1, 0); // already filtered
    const uint32_t sk1_0 = sk2_0 ^ t1;
    if (__builtin_popcount(sk1_0) != leaks[1*4+0]) {
        return 0;
    }

    // Complete checks on K0
    sk0_3 = x0; // already filtered
    sk0_2 = sk1_1 ^ sk1_2;
    if (__builtin_popcount(sk0_2) != leaks[0*4+2]) {
        return 0;
    }
    sk0_1 = sk1_0 ^ sk1_1;
    if (__builtin_popcount(sk0_1) != leaks[0*4+1]) {
        return 0;
    }

    const __m128i X0 = _mm_set_epi32 (0, 0, 0, x0); // already filtered
    __m128i Y0 = _mm_aesenclast_si128(X0, ZERO);
    __m128i T0 = _mm_shuffle_epi8(Y0, INV_SR);
    const uint32_t t0 = (uint32_t)_mm_extract_epi32(T0, 0); // already filtered
    sk0_0 = sk1_0 ^ t0;
    if (__builtin_popcount(sk0_0) != leaks[0*4+0]) {
        return 0;
    }

    // Complete checks on K9
    const __m128i X8 = _mm_set_epi32 (0, 0, 0, sk8_3); // already filtered
    __m128i Y8 = _mm_aesenclast_si128(X8, ZERO);
    __m128i T8 = _mm_shuffle_epi8(Y8, INV_SR);
    const uint32_t t8 = (uint32_t)_mm_extract_epi32(T8, 0); // already filtered
    const uint32_t sk9_0 = sk8_0 ^ t8;
    if (__builtin_popcount(sk9_0) != leaks[9*4+0]) {
        return 0;
    }
    const uint32_t sk9_1 = sk9_0 ^ sk8_1;
    if (__builtin_popcount(sk9_1) != leaks[9*4+1]) {
        return 0;
    }
    const uint32_t sk9_2 = sk9_1 ^ sk8_2;
    if (__builtin_popcount(sk9_2) != leaks[9*4+2]) {
        return 0;
    }
    const uint32_t sk9_3 = sk9_2 ^ sk8_3;
    if (__builtin_popcount(sk9_3) != leaks[9*4+3]) {
        return 0;
    }

    // Complete checks on K10
    const __m128i X9 = _mm_set_epi32 (0, 0, 0, sk9_3); // already filtered
    __m128i Y9 = _mm_aesenclast_si128(X9, ZERO);
    __m128i T9 = _mm_shuffle_epi8(Y9, INV_SR);
    const uint32_t t9 = (uint32_t)_mm_extract_epi32(T9, 0); // already filtered
    const uint32_t sk10_0 = sk9_0 ^ t9;
    if (__builtin_popcount(sk10_0) != leaks[10*4+0]) {
        return 0;
    }
    const uint32_t sk10_1 = sk10_0 ^ sk9_1;
    if (__builtin_popcount(sk10_1) != leaks[10*4+1]) {
        return 0;
    }
    const uint32_t sk10_2 = sk10_1 ^ sk9_2;
    if (__builtin_popcount(sk10_2) != leaks[10*4+2]) {
        return 0;
    }
    const uint32_t sk10_3 = sk10_2 ^ sk9_3;
    if (__builtin_popcount(sk10_3) != leaks[10*4+3]) {
        return 0;
    }

    return 1;
}

int
main()
{
	uint32_t sk[44];
	double leaks[44];

    keySchedule(sk, leaks);

	for (int j = 0; j < 44; ++j) {
		printf("%08x", sk[j]);
		if (((j + 1) % 4) == 0) printf("\n");
	}

	for (int j = 0; j < 44; ++j) {
		printf("%8.0lf", leaks[j]);
		if (((j + 1) % 4) == 0) printf("\n");
	}

    printf("------------------------------\n");
    printf("Step 0: complexity estimation.\n");
    double prod = C_32_k(leaks[4*4+3]);
    printf("[Step 0] Hamming weight: %2.0lf.\n", leaks[4*4+3]);
    printf("[Step 0] Number of key guess for k_4^3: 2^{%.2f}.\n", log2(prod));

    printf("------------------------------\n");
    printf("Step 1: complexity estimation.\n");
    printf("[red]   Hamming weights: %2.0lf, %2.0lf, %2.0lf \n", leaks[0*4+3], leaks[3*4+3], leaks[4*4+2]);
    prod = C_32_k(leaks[0*4+3])
         * C_32_k(leaks[3*4+3])
         * C_32_k(leaks[4*4+2]);
    int size_red = int(prod / pow(2.0,64)) * 2; // multiply by 2 because of the parity of the hw
    printf("[red]   Expected red list size 2^{%.2f}.\n", log2(size_red));

    printf("[green] Hamming weights: %2.0lf, %2.0lf, %2.0lf.\n",
        leaks[7*4+3], leaks[8*4+3], leaks[8*4+2]);
    prod = C_32_k(leaks[7*4+3])
         * C_32_k(leaks[8*4+3])
         * C_32_k(leaks[8*4+2]);
    int size_green = int(prod / pow(2.0,64)); // multiply by 2 because of the parity of the hw, but divide by 2 for the filter on the parity of hw(S[k_8^3])
    printf("[green] Expected green list size 2^{%.2f}.\n", log2(size_green));

    printf("[black] Hamming weights: %2.0lf, %2.0lf, %2.0lf.\n",
        leaks[1*4+3], leaks[5*4+3], leaks[5*4+2]);
    prod = C_32_k(leaks[1*4+3])
         * C_32_k(leaks[5*4+3])
         * C_32_k(leaks[5*4+2]);
    int size_black = prod / pow(2.0,64)*4; // multiply by 4 because of the parity of the hw
    printf("[black] Expected black list size 2^{%.2f} or 0.\n", log2(size_black));
    printf("------------------------------\n");
    printf("Step 2: complexity estimation.\n");
    printf("[Step 2] Complexity of remaining lists generation: 2^{%.2f}.\n", log2(size_red + size_black));
    prod = C_32_k(leaks[6*4+3])
         * C_32_k(leaks[7*4+2])
         * C_32_k(leaks[6*4+1])
         * C_32_k(leaks[8*4+1]);
    auto filter_rem_red = prod / pow(2.0,128)*8; // multiply by 8 because of the parity of the hw
    printf("[rem-red]   Remaining-red list will be of size 2^{%.2f}.\n", log2(size_red) + log2(filter_rem_red));

    prod = C_32_k(leaks[9*4+3])
         * C_32_k(leaks[9*4+2])
         * C_32_k(leaks[9*4+1])
         * C_32_k(leaks[7*4+1]);
    auto filter_rem_black = prod / pow(2.0,128)*32; // multiply by 16 because of the parity of the hw and because blue relation parity filter was already taken into account in green candidates generation. Multiply by another factor 2 because of the condition mod 4 of k_9^1.

    printf("[rem-black] Remaining-black list will be of size 2^{%.2f}.\n", log2(size_black) + log2(filter_rem_black));
    printf("[cart-prod] Complexity of cartesian product: 2^{%.2f}.\n", log2(size_red) + log2(size_black) + log2(filter_rem_red) + log2(filter_rem_black));
    printf("------------------------------\n");

    auto k4_3 = C_32_k(leaks[4*4+3]);
    printf("[*] Step 0: looping over 2^{%.2f} values of key_4^3.\n", log2(k4_3));

    printf("[*] CHEATING: Giving ourselves the correct value of key_4^3...\n");
    for (uint64_t x4_64 = 0; x4_64 < (1ull << 32); ++x4_64) {
        uint32_t x4 = (uint32_t)x4_64;
        if (__builtin_popcount(x4) != leaks[4*4+3]) {
            continue; // X4
        }

        // CHEAT: we give ourselves the value of the last column of the 4th round key
        if (x4 != sk[4*4+3]) {
            continue;
        }

        __m128i X4 = _mm_set_epi8 (
            (x4 >> 16) & 0xff, 0, 0, 0,
            0, (x4 >>  8) & 0xff, 0, 0,
            0, 0, (x4 >>  0) & 0xff, 0,
            0, 0, 0, (x4 >> 24) & 0xff
        );
        __m128i Y4 = _mm_aesenclast_si128(X4, ZERO);
        const uint32_t t4 = (uint32_t)_mm_extract_epi32(Y4, 0);

        if (__builtin_popcount(t4) % 2 != int(leaks[4*5+3] + leaks[4*1+3])% 2) {
            continue; // T4
        }

        printf("[*] Step 1: generating red, green and black candidates.\n");

        vector<triplet_t> candidates_red = getRedCandidates(x4, X4, Y4, leaks);
        printf("[red]   cnt = 2^{%.2f}\n", log2(candidates_red.size()));
        
        vector<triplet_t> candidates_green = getGreenCandidates(x4, X4, Y4, leaks);
        printf("[green] cnt = 2^{%.2f}\n", log2(candidates_green.size()));

        vector<triplet_t> candidates_black = getBlackCandidates(x4, X4, Y4, leaks);
        printf("[black] cnt = 2^{%.2f}\n", log2(candidates_black.size()));

        if (candidates_black.size() == 0) {
            assert(false);
        }

        vector<uint64_t> keys;

        printf("[*] Step 2: Looping over 2^{%.2f} green elements.\n", log2(candidates_green.size()));
        prod = C_32_k(leaks[6*4+3])
             * C_32_k(leaks[7*4+2])
             * C_32_k(leaks[6*4+1])
             * C_32_k(leaks[8*4+1]);
        auto filter_rem_red = prod / pow(2.0,128)*8; // multiply by 8 because of the parity of the hw
        printf("[rem-red]   Remaining-red list will be of size 2^{%.2f}.\n", log2(candidates_red.size()) + log2(filter_rem_red));

        prod = C_32_k(leaks[9*4+3])
             * C_32_k(leaks[9*4+2])
             * C_32_k(leaks[9*4+1])
             * C_32_k(leaks[7*4+1]);
        auto filter_rem_black = prod / pow(2.0,128)*32; // multiply by 16 because of the parity of the hw and because blue relation parity filter was already taken into account in green candidates generation. Multiply by another factor 2 because of the condition mod 4 of k_9^1.

        printf("[rem-black] Remaining-black list will be of size 2^{%.2f}.\n", log2(candidates_black.size()) + log2(filter_rem_black));
        printf("[cart-prod] Cartesian product will loop over 2^{%.2f} elements.\n", log2(candidates_red.size()) + log2(candidates_black.size()) + log2(filter_rem_red) + log2(filter_rem_black));

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < candidates_green.size(); ++i) {
            const triplet_t green = candidates_green[i];
            const uint32_t x4 = green.A;
            const uint32_t x7 = green.B;
            const uint32_t x8 = green.C;

            if ((i & 0xfff) == 0) {
                printf("[*] Progress: 0x%08lx/0x%08lx\n", i, candidates_green.size());
            }
            
            vector<triplet_t> m1;
            vector<triplet_t> m2;

            for (size_t j = 0; j < candidates_red.size(); ++j) {
                const triplet_t red = candidates_red[j];
                const uint32_t x0 = red.A;
                const uint32_t x3 = red.B;

                uint32_t y6 = x3 ^ x7;

                const __m128i Y6 = _mm_set_epi32 (0, 0, 0, y6);
                __m128i X6 = _mm_aesdeclast_si128(Y6, ZERO);
                __m128i T = _mm_shuffle_epi8(X6, SR);
                if (my_popcount(T) != leaks[6*4+3]) {
                    continue; // X6
                }

                const uint32_t x6 = (uint32_t)_mm_extract_epi32(T, 0);
                if (__builtin_popcount(x6 ^ x7) != leaks[7*4+2]) {
                    continue; // C7
                }

                const uint32_t sk8_1 = x6 ^ x8;
                if (__builtin_popcount(sk8_1) != leaks[8*4+1]) {
                    continue; // D8
                }

                const uint32_t sk6_1 = x6 ^ x4;
                if (__builtin_popcount(sk6_1) != leaks[6*4+1]) {
                    continue; // E6
                }

                const triplet_t t = {
                    .A = (uint32_t) x0,
                    .B = (uint32_t) x3,
                    .C = (uint32_t) x6,
                };
                m1.push_back(t);
            }

            const __m128i X8 = _mm_set_epi32 (0, 0, 0, x8);
            __m128i Y8_ = _mm_aesenclast_si128(X8, ZERO);
            __m128i Y8 = _mm_shuffle_epi8(Y8_, INV_SR);
            const uint32_t y8 = (uint32_t)_mm_extract_epi32(Y8, 0); // already filtered

            for (size_t j = 0; j < candidates_black.size(); ++j) {
                const triplet_t black = candidates_black[j];
                const uint32_t x1 = black.A;
                const uint32_t x5 = black.C;

                uint32_t x9 = x5 ^ y8;
                if (__builtin_popcount(x9) != leaks[9*4+3]) {
                    continue; // X9
                }

                if (__builtin_popcount(x8 ^ x9) != leaks[9*4+2]) {
                    continue; // C8
                }

                if (__builtin_popcount(x5 ^ x7) != leaks[7*4+1]) {
                    continue; // D7
                }

                if (__builtin_popcount(x7 ^ x9) != leaks[9*4+1]) {
                    continue; // D9
                }

                const triplet_t t = {
                    .A = (uint32_t) x1,
                    .B = (uint32_t) x5,
                    .C = (uint32_t) x9,
                };
                m2.push_back(t);
            }

            for (size_t k1 = 0; k1 < m1.size(); ++k1) {
                for (size_t k2 = 0; k2 < m2.size(); ++k2) {
                    const triplet_t t1 = m1[k1];
                    const triplet_t t2 = m2[k2];

                    const uint32_t x0 = t1.A;
                    const uint32_t x3 = t1.B;
                    const uint32_t x6 = t1.C;

                    const uint32_t x1 = t2.A;
                    const uint32_t x5 = t2.B;

                    uint32_t sk0_0 = 0;
                    uint32_t sk0_1 = 0;
                    uint32_t sk0_2 = 0;
                    uint32_t sk0_3 = 0;
                    if (finalChecks(leaks, x0, x1, x3, x4, t4, x5, x6, x7, x8, sk0_0, sk0_1, sk0_2, sk0_3)) {
                        printf("Key recovered!! %08x%08x%08x%08x\n", sk0_0, sk0_1, sk0_2, sk0_3);
                    }
                }
            }
        }

        // Free memory
        vector<triplet_t>().swap(candidates_red);
        vector<triplet_t>().swap(candidates_green);
        vector<triplet_t>().swap(candidates_black);

    } // for loop, green list

	return 0;
}
