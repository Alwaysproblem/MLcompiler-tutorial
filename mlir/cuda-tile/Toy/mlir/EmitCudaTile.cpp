#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "toy/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>

using namespace llvm;
using namespace mlir;

namespace {

/// Read file contents as raw bytes.
static FailureOr<std::vector<int8_t>> readFileBytes(StringRef path) {
  auto bufOrErr = MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!bufOrErr)
    return failure();
  auto &buf = *bufOrErr;
  std::vector<int8_t> out(buf->getBufferSize());
  memcpy(out.data(), buf->getBufferStart(), buf->getBufferSize());
  return out;
}

/// Write raw bytes to a file.
static LogicalResult writeFileBytes(StringRef path, ArrayRef<char> bytes) {
  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_None);
  if (ec)
    return failure();
  os.write(bytes.data(), bytes.size());
  os.flush();
  return success();
}

/// Execute external tileiras to assemble tilebc into a binary.
static LogicalResult runTileIRAS(Operation *anchor, StringRef tileirasExe,
                                 StringRef gpuName, StringRef inTilebc,
                                 StringRef outBin) {
  SmallVector<StringRef, 16> args;
  args.push_back(tileirasExe);
  args.push_back("--gpu-name");
  args.push_back(gpuName);
  args.push_back(inTilebc);
  args.push_back("-o");
  args.push_back(outBin);

  std::string errMsg;
  int rc = sys::ExecuteAndWait(tileirasExe, args,
                               /*env=*/std::nullopt,
                               /*redirects=*/{},
                               /*secondsToWait=*/0,
                               /*memoryLimit=*/0, &errMsg);
  if (rc != 0) {
    return anchor->emitError() << "tileiras failed, rc=" << rc << "\n"
                               << errMsg;
  }
  return success();
}

std::error_code createTemporaryFile(SmallVectorImpl<char> &inPath,
                                    StringRef prefix, StringRef suffix) {
  int inFD = -1;
  if (std::error_code ec =
          sys::fs::createTemporaryFile(prefix, suffix, inFD, inPath)) {
    return ec;
  }

  if (std::error_code ec = sys::fs::closeFile(inFD)) {
    return ec;
  }
  return std::error_code();
}

struct EmbedCudaTileBinaryPass
    : public PassWrapper<EmbedCudaTileBinaryPass, OperationPass<ModuleOp>> {

  std::string tileirasExe;
  std::string gpuName;
  std::string cubinOrPtxPath;
  bool useCache;

  EmbedCudaTileBinaryPass(std::string tileirasExe, std::string gpuName,
                          std::string cubinOrPtxPath, bool useCache)
      : tileirasExe(std::move(tileirasExe)), gpuName(std::move(gpuName)),
        cubinOrPtxPath(std::move(cubinOrPtxPath)), useCache(useCache) {}

  void runOnOperation() override {
    ModuleOp top = getOperation();
    MLIRContext *ctx = top.getContext();

    SmallString<256> cudaBinPath;

    top.walk([&](Operation *op) {
      // we assume the MLIR only have one cuda tile module.
      if (op->getName().getStringRef() != "cuda_tile.module")
        return;

      auto cudaMod = dyn_cast<cuda_tile::ModuleOp>(op);
      if (!cudaMod)
        return;

      // ---- Step B: generate tilebc bytes in-process ----
      SmallVector<char, 0> tilebcBytes;
      raw_svector_ostream tilebcOS(tilebcBytes);

      // Using writeBytecode API: writeBytecode(output, moduleOp,
      // BytecodeVersion::kCurrentVersion)
      if (failed(writeBytecode(tilebcOS, cudaMod,
                               cuda_tile::BytecodeVersion::kCurrentVersion))) {
        op->emitError() << "writeBytecode(tilebc) failed";
        signalPassFailure();
        return;
      }

      // ---- Step C: create temp files and invoke tileiras ----
      SmallString<256> inPath;

      if (std::error_code ec =
              createTemporaryFile(inPath, "cuda_tile", "tilebc")) {
        op->emitError() << "failed to create temp in tilebc: " << ec.message();
        signalPassFailure();
        return;
      }

      if (cubinOrPtxPath.empty()) {
        if (std::error_code ec =
                createTemporaryFile(cudaBinPath, "cuda_tile", "bin")) {
          op->emitError() << "failed to create temp out bin: " << ec.message();
          signalPassFailure();
          return;
        }
      } else {
        if (!useCache) {
          if (llvm::sys::fs::exists(cubinOrPtxPath)) {
            op->emitWarning() << "cuda binary file exist  " << cubinOrPtxPath
                              << ", tileiras will overwrite it.";
            std::error_code ec = llvm::sys::fs::remove(cubinOrPtxPath);
            if (ec) {
              op->emitError() << "failed to remove existing cuda binary file: "
                              << ec.message();
              signalPassFailure();
              return;
            }
          }
        }
        cudaBinPath = cubinOrPtxPath;
      }

      if (useCache && llvm::sys::fs::exists(cudaBinPath)) {
        LDBG() << "cuda binary file exist and will be reused: " << cudaBinPath
               << "\n";
        return;
      }

      // ! [FIXME]: please comment out this following code since this is only
      // for testing.
      if (failed(writeFileBytes(inPath, tilebcBytes))) {
        op->emitError() << "failed to write temp tilebc";
        signalPassFailure();
        return;
      }

      if (failed(runTileIRAS(op, tileirasExe, gpuName, inPath, cudaBinPath))) {
        signalPassFailure();
        return;
      }
    });

    LDBG() << "cuda binary path: " << cudaBinPath << "\n";

    top->walk([&](toy::LaunchGpuOp launchOp) {
      // ---- Step D: read cuda binary bytes ----
      auto binBytesOrErr = readFileBytes(cudaBinPath);
      if (failed(binBytesOrErr)) {
        launchOp.emitError() << "failed to read cuda binary file";
        signalPassFailure();
        return;
      }
      auto binBytes = *binBytesOrErr;

      // ---- Step E: embed binary metadata as LaunchGpuOp attributes ----
      // Note: we currently only attach metadata (size/path/arch), not the
      // raw cuda_binary bytes themselves.

      launchOp->setDiscardableAttr(
          "cuda_binary_size",
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                 static_cast<int64_t>(binBytes.size())));
      launchOp->setDiscardableAttr(
          "cuda_binary_path", mlir::StringAttr::get(ctx, cudaBinPath.str()));
      launchOp->setDiscardableAttr("cuda_arch",
                                   mlir::StringAttr::get(ctx, gpuName));
    });

    // ---- Step F: Delete the cuda_tile.module ops ----
    llvm::SmallVector<mlir::Operation *, 32> toErase;
    top->walk([&](cuda_tile::ModuleOp op) { toErase.push_back(op); });

    for (auto op : toErase) {
      op->erase();
    }
  };
};
} // namespace

namespace mlir::toy {

std::unique_ptr<mlir::Pass>
createEmbedCudaTileBinaryPass(std::string tileirasExe, std::string gpuName,
                              std::string cubinOrPtxPath, bool useCache) {
  return std::make_unique<EmbedCudaTileBinaryPass>(tileirasExe, gpuName,
                                                   cubinOrPtxPath, useCache);
};

}; // namespace mlir::toy
