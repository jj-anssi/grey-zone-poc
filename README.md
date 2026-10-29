# PoC attack for the paper "Grey Zone: Side-Channel Attacks Exploiting the AES-128 Key Schedule's 4x32-bit Representation".

This repository is a PoC implementation of the attacks in Section 4.3 of "Grey Zone: Side-Channel Attacks Exploiting the AES-128 Key Schedule's 4x32-bit Representation" published at TCHES 2027 Issue 1. There are two variants of the attack, with different leakage functions. The first variant is the general leakage variant, which considers an arbitrary leakage function (fixed to a random leakage function in the attack). This variant has a parameter OMEGA, which corresponds to the bits of information given by the leakage function on each 32-bit column. The second variant is the Hamming weight variant, which considers that the Hamming weight of each sub-key column leaks.

### Notes on both attacks

The indexes of these implementations differ by 1 compared to the notations of the paper. For instance, the first guessed column is the last column of the subkey k_4 in the implementation, while it is the last column of K_5 in the paper. 

For efficiency purposes and since these attack are PoC, we give ourselves the value of the last column of k_4.

We use AES-NI instructions to get fast Sbox evaluations. In particular, we make use of the two instructions aesenclast and aesdeclast through the two intrinsic functions _mm_aesenclast_si128 and _mm_aesdeclast_si128. We use these two functions to get four parallel SBox evaluations, or four parallel inverse SBox evaluations, to efficiently evaluate the S_rot function. For simplicity, we decided to ignore the constants additions.



### Running the general leakage attack

First compile the attack by specifying the value of OMEGA:
```
make general OMEGA=XX
```
By default, OMEGA is set to 3.33 (estimated complexity after guessing the last column of k_4: around 2^{45} operations), but it is advised to first run with a larger OMEGA (5.0 or 10.0) for faster attacks.

You can then run the attack:
```
./general
```

**Notes**

The attack generates a random affine leakage function, and fixes some leakage thresholds so that the leakage information given by each key column leakage is exactly OMEGA. In real-world attacks, the leakage function may differ from this ideal leakage.



### Running the Hamming weight leakage attack

First compile the attack:
```
make hamming
```
And run the attack:
```
./hamming
```

**Notes**
The C++ implementation of the hamming weight leakage attack has been optimized to run in parallel using OpenMP. 

As detailed in the paper, the time complexity of the full attack reaches about 2^{78.1} operations, which makes it impractical on a standard computer. After giving ourselves the value of the last column of k_4, we validated the attack on a highly parallel server in around 2^{50.4} operations.

In practice, the running time of our optimized implementation of the attack ranges from several hours to several days of CPU time, depending on the observed Hamming weights. Indeed, if we are lucky, the Hamming weights deviate more from the average value of 16, making constructed lists smaller. On the other hand, if most of the Hamming weights are equal to 16, then the complexity will be higher.






