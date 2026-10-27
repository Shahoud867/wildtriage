# Smoke test: MegaDetector + BioCLIP 2

- device: cuda (NVIDIA RTX 4000 Ada Generation), dtype: torch.bfloat16
- images processed: 50/50; with at least one animal box: 48
- BioCLIP 2 zero-shot top-1: 46.0%, top-5: 58.0% (whole-image or best-box crop)
- MegaDetector latency ms: median 226, mean 213 (first image included, includes warm-up)
- BioCLIP 2 latency ms: median 32, mean 37
- peak CUDA memory allocated (GB): 1.5912647247314453

| image | truth | boxes | best score | top-5 |
|---|---|---|---|---|
| S5/U10/U10_R4/S5_U10_R4_IMAG1984 | wildebeest | 6 | 0.87 | buffalo, wildebeest, elephant, cattle, warthog |
| S10/K06/K06_R2/S10_K06_R2_IMAG0671 | warthog | 0 | 0.00 | koribustard, cheetah, topi, warthog, secretarybird |
| S1/D06/D06_R3/S1_D06_R3_PICT0330 | gazellethomsons | 2 | 0.84 | honeybadger, impala, topi, duiker, koribustard |
| S9/I03/I03_R1/S9_I03_R1_IMAG0298 | wildebeest | 21 | 0.88 | wildebeest, hartebeest, eland, topi, elephant |
| S3/N06/N06_R12/S3_N06_R12_IMAG0692 | wildebeest | 16 | 0.86 | wildebeest, warthog, buffalo, elephant, topi |
| S6/U10/U10_R2/S6_U10_R2_IMAG0822 | zebra | 1 | 0.58 | wildcat, genet, elephant, cattle, leopard |
| S2/C08/C08_R2/S2_C08_R2_IMAG0611 | wildebeest | 3 | 0.96 | wildebeest, buffalo, topi, warthog, cattle |
| S7/C08/C08_R2/S7_C08_R2_IMAG2376 | gazellethomsons | 1 | 0.88 | wildcat, aardwolf, secretarybird, koribustard, reedbuck |
| S7/U10/U10_R1/S7_U10_R1_IMAG0205 | wildebeest | 1 | 0.97 | wildebeest, warthog, eland, topi, elephant |
| SER_S11/I03/I03_R1/SER_S11_I03_R1_IMAG0301 | gazellethomsons | 1 | 0.33 | giraffe, hartebeest, reedbuck, topi, koribustard |
| S9/N06/N06_R3/S9_N06_R3_IMAG0325 | gazellethomsons | 1 | 0.12 | honeybadger, topi, impala, reedbuck, eland |
| S4/D06/D06_R2/S4_D06_R2_IMAG0041 | secretarybird | 2 | 0.83 | secretarybird, genet, honeybadger, guineafowl, serval |
| S9/K06/K06_R2/S9_K06_R2_IMAG0159 | gazellethomsons | 2 | 0.56 | reedbuck, impala, dikdik, duiker, caracal |
| S8/N09/N09_R2/S8_N09_R2_IMAG0033 | zebra | 5 | 0.63 | eland, hippopotamus, zebra, warthog, topi |
| S9/I03/I03_R1/S9_I03_R1_IMAG1032 | wildebeest | 5 | 0.58 | cattle, bat, porcupine, fire, hyenaspotted |
| S9/C08/C08_R3/S9_C08_R3_IMAG0042 | guineafowl | 2 | 0.91 | guineafowl, dikdik, monkeyvervet, secretarybird, honeybadger |
| S8/C01/C01_R1/S8_C01_R1_IMAG2423 | zebra | 4 | 0.57 | warthog, elephant, eland, waterbuck, wildebeest |
| S4/N06/N06_R2/S4_N06_R2_IMAG1008 | wildebeest | 2 | 0.70 | civet, waterbuck, porcupine, wildcat, genet |
| SER_S11/I03/I03_R2/SER_S11_I03_R2_IMAG1111 | gazellethomsons | 1 | 0.12 | duiker, topi, wildcat, eland, secretarybird |
| S2/D06/D06_R3/S2_D06_R3_IMAG0303 | wildebeest | 15 | 0.96 | wildebeest, buffalo, warthog, waterbuck, elephant |
| S9/E06/E06_R1/S9_E06_R1_IMAG2953 | gazellethomsons | 3 | 0.87 | impala, hartebeest, reedbuck, topi, eland |
| S2/K06/K06_R3/S2_K06_R3_PICT0906 | wildebeest | 4 | 0.87 | wildebeest, eland, waterbuck, buffalo, cattle |
| S9/D03/D03_R1/S9_D03_R1_IMAG1159 | hippopotamus | 1 | 0.97 | hippopotamus, warthog, elephant, eland, giraffe |
| S10/I03/I03_R3/S10_I03_R3_IMAG0545 | wildebeest | 4 | 0.34 | buffalo, wildebeest, cattle, elephant, topi |
| S1/E06/E06_R4/S1_E06_R4_PICT1884 | baboon | 2 | 0.86 | baboon, warthog, aardvark, wildebeest, waterbuck |
| S9/N09/N09_R2/S9_N09_R2_IMAG0164 | wildebeest | 2 | 0.86 | wildebeest, warthog, hartebeest, elephant, eland |
| S7/U10/U10_R1/S7_U10_R1_IMAG0026 | otherbird | 1 | 0.95 | koribustard, secretarybird, honeybadger, elephant, ostrich |
| S10/I03/I03_R3/S10_I03_R3_IMAG0686 | wildebeest | 8 | 0.67 | wildebeest, cattle, warthog, buffalo, eland |
| S2/C08/C08_R2/S2_C08_R2_IMAG0838 | wildebeest | 2 | 0.84 | wildebeest, warthog, hartebeest, waterbuck, topi |
| S4/I03/I03_R2/S4_I03_R2_IMAG0138 | buffalo | 1 | 0.97 | buffalo, wildebeest, cattle, topi, warthog |
| S7/C01/C01_R3/S7_C01_R3_IMAG7114 | warthog | 4 | 0.54 | warthog, wildcat, cattle, buffalo, elephant |
| S5/D06/D06_R1/S5_D06_R1_IMAG3896 | wildebeest | 1 | 0.63 | wildebeest, buffalo, hartebeest, eland, warthog |
| S9/K06/K06_R2/S9_K06_R2_IMAG0693 | gazellethomsons | 6 | 0.27 | impala, giraffe, koribustard, eland, cheetah |
| S3/U10/U10_R12/S3_U10_R12_IMAG0053 | wildebeest | 2 | 0.96 | wildebeest, warthog, hartebeest, topi, buffalo |
| S2/E06/E06_R3/S2_E06_R3_IMAG2888 | wildebeest | 14 | 0.96 | wildebeest, warthog, eland, elephant, waterbuck |
| S5/U10/U10_R4/S5_U10_R4_IMAG1107 | wildebeest | 9 | 0.76 | wildebeest, warthog, eland, waterbuck, elephant |
| S8/C08/C08_R1/S8_C08_R1_IMAG0085 | gazellethomsons | 4 | 0.97 | impala, dikdik, topi, eland, honeybadger |
| S9/K06/K06_R1/S9_K06_R1_IMAG1582 | gazellethomsons | 2 | 0.66 | koribustard, serval, secretarybird, cheetah, leopard |
| S10/K06/K06_R2/S10_K06_R2_IMAG0055 | gazellethomsons | 0 | 0.00 | topi, cheetah, koribustard, impala, dikdik |
| S7/C08/C08_R1/S7_C08_R1_IMAG0415 | wildebeest | 1 | 0.83 | buffalo, elephant, wildebeest, cattle, eland |
| S8/E06/E06_R3/S8_E06_R3_IMAG2110 | zebra | 2 | 0.31 | wildebeest, hippopotamus, topi, giraffe, eland |
| S9/N06/N06_R2/S9_N06_R2_IMAG0562 | wildebeest | 3 | 0.59 | wildebeest, elephant, topi, hartebeest, eland |
| SER_S11/N09/N09_R2/SER_S11_N09_R2_IMAG0351 | hartebeest | 1 | 0.89 | buffalo, cattle, secretarybird, reedbuck, wildebeest |
| S7/N09/N09_R2/S7_N09_R2_IMAG0343 | gazellegrants | 1 | 0.52 | dikdik, reedbuck, secretarybird, waterbuck, aardwolf |
| S9/E06/E06_R2/S9_E06_R2_IMAG1359 | zebra | 2 | 0.41 | zebra, hyenastriped, wildebeest, eland, warthog |
| S2/K06/K06_R3/S2_K06_R3_PICT0370 | wildebeest | 4 | 0.90 | wildebeest, warthog, hartebeest, buffalo, elephant |
| S9/D03/D03_R3/S9_D03_R3_IMAG0088 | zebra | 1 | 0.88 | honeybadger, secretarybird, warthog, wildebeest, giraffe |
| S9/C08/C08_R2/S9_C08_R2_IMAG0520 | zebra | 2 | 0.75 | zebra, hyenastriped, wildebeest, eland, elephant |
| S10/I03/I03_R3/S10_I03_R3_IMAG0822 | wildebeest | 2 | 0.57 | cattle, buffalo, wildebeest, eland, topi |
| S9/D03/D03_R2/S9_D03_R2_IMAG0492 | zebra | 11 | 0.95 | wildebeest, eland, giraffe, impala, warthog |
