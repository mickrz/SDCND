/*
 * mikeDebug.h
 *
 *  Created on: July 5, 2017
 *  Author: M. Rzucidlo
 *
 */


#ifndef MIKE_DEBUG_H_
#define MIKE_DEBUG_H_
using Eigen::IOFormat;

const float MICROSECONDS_TO_SECONDS = 1000000.0;
const float EPSILON = 0.0001;

/** Use this with discretion or a GPU preferably since this causes the app to run very slowly... */
//#define EXTRA_DEBUG
#define MINIMAL_DEBUG

/** First argument is -2 which is a hack. It represents FullPrecision, but I
    could not resolve the build error of it not being declared in scope. */
const static IOFormat HeavyFmt(-2, 0, ", ", ";\n", "[", "]", "[", "]");

#define PRINTFUNC(LEVEL, FILE, LINE, FUNCTION, TEXT) \
{ \
  std::cout << LEVEL << ": " << FILE << ", " << LINE << ", " << FUNCTION << " - " << TEXT << std::endl; \
}  

#define ERROR(FILE, LINE, FUNCTION, TEXT) \
{ \
  PRINTFUNC("ERROR: ", FILE, LINE, FUNCTION, TEXT) \
}

#define INFO(FILE, LINE, FUNCTION, TEXT) \
{ \
  PRINTFUNC("INFO: ", FILE, LINE, FUNCTION, TEXT) \
}

#define PRINTMATRIX(FILE, LINE, FUNCTION, NAME, MATRIX) \
{ \
  std::cout << FILE << ", " << LINE << ", " << FUNCTION << ", " << NAME << ":\n" << MATRIX.format(HeavyFmt) << std::endl; \
}

#endif /* MIKE_DEBUG_H_ */