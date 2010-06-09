/*float gasdev(long *idum) ;
float gauss(long *seed, float mean, float sigma) ;
float nrran0(long *idum) ;
float nrran1(long *idum) ;
float nrran2(long *idum) ;
*/

/*
 * typedef
 */
typedef struct {
        int           index;
        float         power;
        float         mean;
} outputStruct; 

int output_spectrum(outputStruct *,int,int);
/*int init_output_file();
void terminate_output_file();
void output_spectrum_to_file(outputStruct *);

extern "C" void do_analyze_on_gpu();
*/
