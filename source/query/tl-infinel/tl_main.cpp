#include "tl_defines.h"
#include <ash/size.h>
#include <ash/stop_watch.h>
#include <ash/utility/dbg_log.h>
#include <gstream/cuda_proxy.h>
#include <gstream/grid_format/grid_stream2.h>
#include <gstream/qrydef.h>
#include <ash/utility/dbg_log.h>

program_args make_args(char* argv[]) {
    using namespace gstream;
    program_args args;
    args.graph_name = argv[1];
    args.graph_dir = argv[2];
    args.device_buffer_size = ash::MiB(strtoull(argv[3], nullptr, 10));
    args.chunk_num = strtoull(argv[4], nullptr, 10);
    args.chunk_size = strtoull(argv[5], nullptr, 10);
    if (std::tolower(argv[6][0]) == 'y')
        args.load_balance_mode = true;
    else if (std::tolower(argv[6][0]) == 'n')
        args.load_balance_mode = false;
    else
        ASH_FATAL("Parameter $6: Load balance mode flag is wrong");
    if (std::tolower(argv[7][0]) == 'y')
        args.double_buffering_mode = true;
    else if (std::tolower(argv[7][0]) == 'n')
        args.double_buffering_mode = false;
    else
        ASH_FATAL("Parameter $7: Double buffering mode flag is wrong");
    if (std::tolower(argv[8][0]) == 'y')
        args.only_kernel_mode = true;
    else if (std::tolower(argv[8][0]) == 'n')
        args.only_kernel_mode = false;
    else
        ASH_FATAL("Parameter $8: Only Kernel mode flag is wrong");
    if (std::tolower(argv[9][0]) == 'y')
        args.verification_mode = true;
    else if (std::tolower(argv[9][0]) == 'n')
        args.verification_mode = false;
    else
        ASH_FATAL("Parameter $9: Verification mode flag is wrong");

    int num_devices;
    bool const r = cuda::device_count(&num_devices);
    if (!r || num_devices == 0)
        ASH_FATAL("No CUDA devices!\n");
    return args;
}


int main(int const argc, char* argv[]) {
    using namespace gstream;
    if (argc < 10) {
        fprintf(stderr,
            "usage %s "
            "${GRAPH_NAME} ${GRAPH_PATH} "
            "${GPU_MEMORY_BUF_SIZE:MB} "
            "${CHUNK_NUM} ${CHUNK_SIZE:BYTE} "
            "${LOAD_BALANCE_MODE_FLAG} ${DOUBLE_BUFFERING_MODE_FLAG}  ${ONLY_KERNEL_MODE_FLAG} ${VERIFICATION_MODE_FLAG}\n",
            argv[0]
        );
        return -1;
    }

    program_args const args = make_args(argv);

    grid_format::grid_stream2 gs;
    // Init a grid stream
    {
        grid_format::grid_stream2::config_t gs_cfg;
        gs_cfg.graph_dir   = args.graph_dir;
        gs_cfg.graph_name  = args.graph_name;
        gs_cfg.stream_opt  = 0x01;
        if (!gs.open(gs_cfg))
            ASH_FATAL("Please check that the files %s.graph_info and %s.graph exist in your %s path.\n", args.graph_name, args.graph_name, args.graph_dir);
    }

    {
        // init
        ASH_LOG("initializing...");
        init(&args);
        gstream_query tl_query;
        init_tl_query(&tl_query, &gs, &args);
        gstream_framework* tl_fw = init_framework(&tl_query);

        // query
        ASH_LOG("query start...");
        ash::stop_watch watch;
        begin_tl(tl_fw);
        ASH_LOG("Elapsed time: %lf sec.", watch.elapsed_sec());
        
        // finalize
        ASH_LOG("finalizing...");
        cleanup_framework(tl_fw);
        cleanup();
        ASH_LOG("end");
    }
    
    
    return 0;
}
