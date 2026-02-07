#pragma once
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>

namespace vega_rt {
    inline void init_logging(int argc, char** argv) {
        google::InitGoogleLogging(argv[0]);
        FLAGS_logtostderr = 1;
        FLAGS_minloglevel = 0;  // 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
        FLAGS_stderrthreshold = 0;  // 输出所有级别到 stderr
        FLAGS_colorlogtostderr = 1;
    }
}