#ifndef POW2_H
#define POW2_H

#include <memory>

namespace mlir {

class Pass;
#define GEN_PASS_DECL_POW2PASS
#include "Pow2Pass.inc"

namespace mhlo {

std::unique_ptr<Pass> createSubstitutePow2Pass();
std::unique_ptr<Pass> createSubstitutePow2Pass(const Pow2PassOptions &options);
std::unique_ptr<mlir::Pass> createStaticOpCounter();

#define GEN_PASS_REGISTRATION
#include "Pow2Pass.inc"

} // namespace mhlo
} // namespace mlir

#endif // POW2_H
