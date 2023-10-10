#ifndef TANH_H
#define TANH_H

#include <memory>

namespace mlir {
class Pass;

namespace mhlo {

std::unique_ptr<Pass> createPopulateTanhPass();

} // namespace mhlo
} // namespace mlir

#endif // TANH_H
