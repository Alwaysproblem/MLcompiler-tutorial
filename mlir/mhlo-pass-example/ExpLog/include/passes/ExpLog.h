#ifndef TANH_H
#define TANH_H

#include <memory>

namespace mlir {
class Pass;

namespace mhlo {

std::unique_ptr<Pass> createExpLogEmitPass();

} // namespace mhlo
} // namespace mlir

#endif // TANH_H
