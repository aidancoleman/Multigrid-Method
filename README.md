CASE STUDY 4 COMPILE AND RUN ON SEAGULL INSTRUCTIONS

cs4_1.c = First task
#COMPILE
gcc -std=c99 -O3 -Wall -Wextra -o cs4_1 cs4_1.c -lm
#RUN 
./cs4_1

cs4_2.c = Second task part 1
#COMPILE 
gcc -std=c99 -O3 -Wall -Wextra cs4_2.c -o cs4_2 -lm
#RUN for lmax = 7 for example (levels of coarseness)
./cs4_2 7 

cs4_3.c = Second task part 2
#COMPILE 
gcc -O3 -std=c99 cs4_3.c -o cs4_3 -lm
#RUN
./cs4_3
