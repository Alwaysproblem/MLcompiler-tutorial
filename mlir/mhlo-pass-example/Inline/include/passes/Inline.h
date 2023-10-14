#ifndef POW2_H
#define POW2_H

#include <memory>

namespace mlir {
class Pass;

namespace mhlo {

std::unique_ptr<Pass> createInlinePass();

} // namespace mhlo
} // namespace mlir

#endif // POW2_H
