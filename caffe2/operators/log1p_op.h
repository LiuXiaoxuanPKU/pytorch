#ifndef CAFFE2_OPERATORS_LOG1P_OP_H_
#define CAFFE2_OPERATORS_LOG1P_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct Log1pFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Log1p(N, X, Y, context);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOG1P_OP_H_

