#pragma once

#ifdef GEO_OPTIX

#ifndef M_OPTIX_UTILS
#define M_OPTIX_UTILS

#include <sstream>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#define OPTIX_STRINGIFY2(name) #name
#define OPTIX_STRINGIFY(name) OPTIX_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME OPTIX_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
#define OPTIX_SAMPLE_DIR OPTIX_STRINGIFY(OPTIX_SAMPLE_DIR_DEFINE)

#define OPTIX_CHECK_THROW( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw std::runtime_error( ss.str().c_str());                   \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_THROW_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            throw std::runtime_error(ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )
    
#endif  // M_OPTIX_UTILS

#endif 