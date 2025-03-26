#pragma once

#ifdef GEO_OPTIX

#ifndef PROGRAM_H
#define PROGRAM_H

#include<utils/gpumemory.cuh>
#include<geometry/triangle.cuh>
#include<utils/m_optix_utils.cuh>

#include<iostream>
#include<iomanip>
#include<fstream>

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}


static bool read_ptx_file(std::string &str, const std::string file_name){
    // Try to open file
    std::ifstream file( file_name.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

template<typename T>
struct SbtRecord{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template<typename T>
class Program{
    public:
        Program(const char *data, size_t size, OptixDeviceContext optix){
            char log[2048];
            size_t sizeof_log = sizeof(log);

            OptixModule optix_module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                // Default options for module
                OptixModuleCompileOptions module_compile_options = {};

                // Pipeline options must be consistent for all modules used in a single pipeline
                pipeline_compile_options.usesMotionBlur = false;
                pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipeline_compile_options.numPayloadValues = 3;
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
                OPTIX_CHECK_THROW_LOG(optixModuleCreateFromPTX(
                        optix,
                        &module_compile_options,
                        &pipeline_compile_options,
                        data,
                        size,
                        log,
                        &sizeof_log,
                        &optix_module
                ));
            }

            OptixProgramGroup raygen_prog_group = nullptr;
            OptixProgramGroup miss_prog_group = nullptr;
            OptixProgramGroup hitgroup_prog_group = nullptr;
            {
                OptixProgramGroupOptions program_group_options = {};
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                    
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = optix_module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                    optix,
                    &raygen_prog_group_desc,
                    1,
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group
                ));

                OptixProgramGroupDesc miss_prog_group_desc  = {};
                miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module            = optix_module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                    optix,
                    &miss_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &miss_prog_group
                ));

                OptixProgramGroupDesc hitgroup_prog_group_desc = {};
                hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroup_prog_group_desc.hitgroup.moduleCH            = optix_module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                    optix,
                    &hitgroup_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &hitgroup_prog_group
                ));
            }

            // Linking
            {
                const uint32_t max_trace_depth = 1;
                OptixProgramGroup program_groups[] = {
                    raygen_prog_group,
                    miss_prog_group,
                    hitgroup_prog_group
                };

                OptixPipelineLinkOptions pipeline_link_options = {};
                pipeline_link_options.maxTraceDepth = max_trace_depth;
                pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

                OPTIX_CHECK_THROW_LOG(optixPipelineCreate(
                    optix,
                    &pipeline_compile_options,
                    &pipeline_link_options,
                    program_groups,
                    sizeof(program_groups) / sizeof(program_groups[0]),
                    log,
                    &sizeof_log,
                    &m_pipeline
                ));

                OptixStackSizes stack_sizes = {};
                for(auto &prog: program_groups){
                    OPTIX_CHECK_THROW(optixUtilAccumulateStackSizes(prog, &stack_sizes));
                }

                uint32_t direct_callable_stack_size_from_traversal;
                uint32_t direct_callable_stack_size_from_state;
                uint32_t continuation_stack_size;
                OPTIX_CHECK_THROW(optixUtilComputeStackSizes(
                    &stack_sizes, max_trace_depth,
                    0,
                    0,
                    &direct_callable_stack_size_from_traversal,
                    &direct_callable_stack_size_from_state,
                    &continuation_stack_size
                ));
            }

            // Shader binding table
            {
                CUdeviceptr raygen_record;
                const size_t raygen_record_size = sizeof(SbtRecord<typename T::RayGenData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
                SbtRecord<typename T::RayGenData> rg_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

                CUDA_CHECK_THROW(cudaMemcpy(
                    reinterpret_cast<void*>(raygen_record),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                ));

                CUdeviceptr miss_record;
                size_t miss_record_size = sizeof(SbtRecord<typename T::MissData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
                SbtRecord<typename T::MissData> ms_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
                CUDA_CHECK_THROW(cudaMemcpy(
                    reinterpret_cast<void*>(miss_record),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
                ));

                CUdeviceptr hitgroup_record;
                size_t hitgroup_record_size = sizeof(SbtRecord<typename T::HitGroupData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
                SbtRecord<typename T::HitGroupData> hg_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
                CUDA_CHECK_THROW(cudaMemcpy(
                    reinterpret_cast<void*>(hitgroup_record),
                    &hg_sbt,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice
                ));

                m_sbt.raygenRecord = raygen_record;
                m_sbt.missRecordBase = miss_record;
                m_sbt.missRecordStrideInBytes = sizeof(SbtRecord<typename T::MissData>);
                m_sbt.missRecordCount             = 1;
                m_sbt.hitgroupRecordBase          = hitgroup_record;
                m_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<typename T::HitGroupData>);
                m_sbt.hitgroupRecordCount         = 1;
            }
        }

        void invoke(const typename T::Params &params, const uint3 &dim, cudaStream_t stream){
            CUDA_CHECK_THROW(cudaMemcpyAsync(m_params_gpu.ptr(), &params, sizeof(typename T::Params), cudaMemcpyHostToDevice, stream));
			OPTIX_CHECK_THROW(optixLaunch(
                m_pipeline, stream, 
                (CUdeviceptr)(uintptr_t)m_params_gpu.ptr(), 
                sizeof(typename T::Params), 
                &m_sbt, 
                dim.x, dim.y, dim.z
            ));
        }


    private:
        OptixShaderBindingTable m_sbt = {};
        OptixPipeline m_pipeline = nullptr;
        GPUVector<typename T::Params> m_params_gpu = GPUVector<typename T::Params>(1);
};

OptixDeviceContext g_optix;

static bool initialize_optix(){
    static bool ran_before = false;
    static bool is_optix_initialized = false;
    if(ran_before)return is_optix_initialized;

    ran_before = true;

    // Initialize CUDA with a no-op call to the the CUDA runtime API
	CUDA_CHECK_THROW(cudaFree(nullptr));
	try {
		// Initialize the OptiX API, loading all API entry points
		OPTIX_CHECK_THROW(optixInit());
		// Specify options for this context. We will use the default options.
		OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
		// Associate a CUDA context (and therefore a specific GPU) with this
		// device context
		CUcontext cuCtx = 0; // NULL means take the current active context
		OPTIX_CHECK_THROW(optixDeviceContextCreate(cuCtx, &options, &g_optix));
	} catch (std::exception& e) {
		std::cout << "OptiX failed to initialize: " << std::endl;
		return false;
	}
	is_optix_initialized = true;
	return true;
}

class Gas {
public:
	Gas(GPUVector<Triangle>& triangles, OptixDeviceContext optix, cudaStream_t stream) {
		// Specify options for the build. We use default options for simplicity.
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Populate the build input struct with our triangle data as well as
		// information about the sizes and types of our data
		const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		OptixBuildInput triangle_input = {};

		CUdeviceptr d_triangles = (CUdeviceptr)(uintptr_t)triangles.ptr();

		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices = (uint32_t)triangles.size()*3;
		triangle_input.triangleArray.vertexBuffers = &d_triangles;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		// Query OptiX for the memory requirements for our GAS
		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK_THROW(optixAccelComputeMemoryUsage(optix, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

		// Allocate device memory for the scratch space buffer as well
		// as the GAS itself
		GPUVector<char> gas_tmp_buffer(gas_buffer_sizes.tempSizeInBytes);
		m_gas_gpu_buffer.resize(gas_buffer_sizes.outputSizeInBytes);
		OPTIX_CHECK_THROW(optixAccelBuild(
			optix,
			stream,
			&accel_options,
			&triangle_input,
			1,           // num build inputs
			(CUdeviceptr)(uintptr_t)gas_tmp_buffer.ptr(),
			gas_buffer_sizes.tempSizeInBytes,
			(CUdeviceptr)(uintptr_t)m_gas_gpu_buffer.ptr(),
			gas_buffer_sizes.outputSizeInBytes,
			&m_gas_handle, // Output handle to the struct
			nullptr,       // emitted property list
			0              // num emitted properties
		));
	}
	OptixTraversableHandle handle() const {
		return m_gas_handle;
	}
private:
	OptixTraversableHandle m_gas_handle;
	GPUVector<char> m_gas_gpu_buffer;
};


#endif  // NDEF PROGRAM_H

#endif  // GEO_OPTIX