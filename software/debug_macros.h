#include <stdio.h>

//if DEBUG is on then replace debugging directives with real code
#ifdef DEBUG
#define debug_perror(string) perror(string);
#define debug_fprintf(stream, ...) fprintf(stream,  __VA_ARGS__);

//otherwise just ignore
#else
#define debug_perror(string)
#define debug_fprintf(stream, ...)

#endif

