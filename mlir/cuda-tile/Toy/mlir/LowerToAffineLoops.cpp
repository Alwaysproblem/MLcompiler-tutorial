//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "cuda_shim/CudaShimBuilder.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DebugLog.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = memref::AllocOp::create(rewriter, loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder and the range of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn =
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter and the loop
        // induction variables. This function will return the value to store at
        // the current index.
        Value valueToStore = processIteration(nestedBuilder, ivs);
        affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                      ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      // Generate loads for the element of 'lhs' and 'rhs' at the
      // inner loop.
      auto loadedLhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
      auto loadedRhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

      // Create the binary operation performed on the loaded
      // values.
      return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs);
    });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        affine::AffineStoreOp::create(
            rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                           op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine Conversion Patterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      Value input = adaptor.getInput();

      // Transpose the elements by generating a load from the
      // reverse indices.
      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
      return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: MatMul operations
//===----------------------------------------------------------------------===//

struct MatMulOpLowering : public ConversionPattern {
  MatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::MatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    RankedTensorType lhsType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    RankedTensorType rhsType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    auto tensorType =
        llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));

    auto elemType = llvm::dyn_cast<FloatType>(tensorType.getElementType());

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank() + 1, /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank() + 1, /*Value=*/1);
    SmallVector<int64_t, 4> upperBounds{lhsShape[0], rhsShape[0], rhsShape[1]};

    // add initialization of result tensor.
    // Create a nest of affine loops to initialize the result tensor to 0.
    affine::buildAffineLoopNest(
        rewriter, loc, {0, 0}, tensorType.getShape(), {1, 1},
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Create a constant float value of 0.0.
          auto valueToStore = arith::ConstantFloatOp::create(
              nestedBuilder, loc, elemType,
              llvm::APFloat::getZero(elemType.getFloatSemantics()));

          // Store the constant value into the allocated memory.
          affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                        ivs);
        });

    // Create a nest of affine loops for matrix multiplication.
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Extract loop induction variables.
          Value m = ivs[0];
          Value k = ivs[1];
          Value n = ivs[2];

          // Create an adaptor for the remapped operands of the MatMulOp.
          toy::MatMulOpAdaptor matmulAdaptor(operands);

          // Load elements from the left-hand side and right-hand side matrices.
          auto loadedLhs = affine::AffineLoadOp::create(
              nestedBuilder, loc, matmulAdaptor.getLhs(), ValueRange{m, k});

          auto loadedRhs = affine::AffineLoadOp::create(
              nestedBuilder, loc, matmulAdaptor.getRhs(), ValueRange{k, n});
          // Load elements from the result tensor from initial process above.
          auto loadedRes = affine::AffineLoadOp::create(
              nestedBuilder, loc, alloc, ValueRange{m, n});

          // Perform the multiplication and addition operations.
          auto mulop =
              arith::MulFOp::create(nestedBuilder, loc, loadedLhs, loadedRhs);
          auto valueToStore =
              arith::AddFOp::create(nestedBuilder, loc, loadedRes, mulop);

          // Store the result back into the allocated memory.
          affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                        ValueRange{m, n});
        });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

memref::GlobalOp createGlobalForStringAttr(mlir::PatternRewriter &rewriter,
                                           Operation *op,
                                           llvm::StringRef sym_name,
                                           StringAttr attr) {
  auto loc = op->getLoc();
  auto moduleOp = op->getParentOfType<ModuleOp>();

  if (auto global = moduleOp.lookupSymbol<memref::GlobalOp>(sym_name); global) {
    return global;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto str = attr.getValue();
  std::vector<uint8_t> bytes(str.begin(), str.end());
  bytes.push_back(0);

  auto type = RankedTensorType::get({(int64_t)bytes.size()},
                                    rewriter.getIntegerType(8));

  auto memrefType =
      MemRefType::get({(int64_t)bytes.size()}, rewriter.getIntegerType(8));

  auto denseAttr = DenseElementsAttr::get(type, llvm::ArrayRef<uint8_t>(bytes));

  auto global = memref::GlobalOp::create(
      rewriter, loc, sym_name,
      /*sym_visibility=*/rewriter.getStringAttr("private"), memrefType,
      denseAttr,
      /*constant=*/true,
      /*alignment=*/nullptr);

  return global;
}

arith::IndexCastOp getIndexFromValue(mlir::PatternRewriter &rewriter,
                                     Location loc, Value value) {
  auto extractOp = memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, loc, rewriter.getIndexType(), value);
  auto indexCastOp = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI64Type(), extractOp.getResult());
  return indexCastOp;
}

arith::IndexCastOp getIndexFromGlobalMemref(mlir::PatternRewriter &rewriter,
                                            Location loc,
                                            memref::GlobalOp global) {

  auto getGlobalOp = memref::GetGlobalOp::create(
      rewriter, loc, global.getType(), global.getName());

  return getIndexFromValue(rewriter, loc, getGlobalOp.getResult());
}

func::CallOp
createCallToCudaShimMalloc(mlir::PatternRewriter &rewriter, Location loc,
                           CudaShimRegistry &registry, func::CallOp stream,
                           arith::ConstantIntOp nbytesVal, bool isHostShared) {
  arith::ConstantIntOp isHostSharedVal;
  if (isHostShared) {
    isHostSharedVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  } else {
    isHostSharedVal = arith::ConstantIntOp::create(rewriter, loc, 0, 1);
  }
  auto sreamVal = stream.getResult(0);
  auto callee = registry.call(rewriter, stream, CudaShimFn::Malloc,
                              ValueRange{nbytesVal, sreamVal, isHostSharedVal});
  return callee;
}

unsigned long getNbytes(Type tensorType) {
  auto ranked_tensor_type = llvm::cast<MemRefType>(tensorType);
  return llvm::divideCeil(ranked_tensor_type.getNumElements() *
                              ranked_tensor_type.getElementTypeBitWidth(),
                          8);
}

struct LanchGpuLowering : public OpConversionPattern<toy::LaunchGpuOp> {
  using OpConversionPattern<toy::LaunchGpuOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::LaunchGpuOp launchGpuOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = launchGpuOp->getLoc();
    CudaShimRegistry registry(launchGpuOp->getParentOfType<ModuleOp>());

    for (auto ranked_tensor_type : launchGpuOp->getOperands()) {
      if (!llvm::isa<RankedTensorType>(ranked_tensor_type.getType())) {
        return rewriter.notifyMatchFailure(launchGpuOp,
                                           "expected operand to be a "
                                           "ranked tensor type");
      }
    }

    // %3 = toy.launch_gpu @outlined_gpu_kernel_0(%0, %2, %1) {cuda_arch =
    // "sm_120", cuda_binary_path = "/tmp/cuda_tile-d7f3fd.bin",
    // cuda_binary_size = 10112 : i64, grid = array<i64: 1, 1, 1>} :
    // (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    // the `%3` is the output value of the launch op, `%0`, `%2`, `%1` are the
    // operands to be passed to the GPU kernel, and the attributes are the
    // launch configuration for the GPU kernel.
    // so we need to create the output tensor since the cuda tile entry op
    // will take the output tensor as argument instead of return value.
    // and as we did before, the last operand is the output tensor, and the rest
    // are input tensors.
    // moreover, we assume the memory life time of the output tensor is the
    // whole main function, so we can just allocate it at the beginning of the
    // main function and pass it to the cuda tile entry op which means we won't
    // promote the deallocation of the output tensor from the last op of the
    // block to the end of use.

    auto outputType = llvm::cast<RankedTensorType>(
        launchGpuOp->getResults().front().getType());

    auto outputMemRefType = convertTensorToMemRef(outputType);
    auto outputTensorAlloc =
        insertAllocAndDealloc(outputMemRefType, loc, rewriter);

    // extract the `cuda_binary_path` attribute from the launch op, and create a
    // global memref for it, which will be used in the cuda tile entry op to
    // load the cuda binary.
    auto cudaBinaryPathAttr =
        launchGpuOp->getDiscardableAttr("cuda_binary_path");
    if (!cudaBinaryPathAttr) {
      return rewriter.notifyMatchFailure(
          launchGpuOp, "expected 'cuda_binary_path' attribute to be present");
    }

    auto cudaBinaryPathStr = llvm::dyn_cast<StringAttr>(cudaBinaryPathAttr);
    if (!cudaBinaryPathStr) {
      return rewriter.notifyMatchFailure(
          launchGpuOp, "expected 'cuda_binary_path' attribute to be a string");
    }

    // add the global memref for the cuda binary path and the kernel name.
    auto cuda_blob_memref = createGlobalForStringAttr(
        rewriter, launchGpuOp, "cuda_blob", cudaBinaryPathStr);

    auto kernelName = launchGpuOp.getCallee();

    auto kernel_name_memref = createGlobalForStringAttr(
        rewriter, launchGpuOp, "kname", rewriter.getStringAttr(kernelName));

    // load the cuda binary path from the global memref.
    auto cuda_blob_index =
        getIndexFromGlobalMemref(rewriter, loc, cuda_blob_memref);
    auto kname_loaded_index =
        getIndexFromGlobalMemref(rewriter, loc, kernel_name_memref);

    // Added blob size.
    auto blob_size =
        llvm::cast<MemRefType>(cuda_blob_memref.getType()).getShape()[0];
    auto blob_size_index =
        arith::ConstantIntOp::create(rewriter, loc, blob_size, 64);

    // create a call to the cuda shim function to load the cuda binary
    auto load_cubin_callee =
        registry.call(rewriter, launchGpuOp, CudaShimFn::LoadModuleFromFile,
                      ValueRange{cuda_blob_index, blob_size_index});

    // create a stream for the kernel launch, for simplicity we use the default
    // stream (0).
    auto stream =
        registry.call(rewriter, launchGpuOp, CudaShimFn::StreamCreate);

    // we assume the number of output tensors is only 1, and it's the last
    // operand of the launch op.
    llvm::SmallVector<Value, 8> devicePtrs;
    llvm::SmallVector<Value, 8> cudaAllInputs;

    for (auto operand : adaptor.getOperands()) {
      cudaAllInputs.push_back(operand);
    }
    cudaAllInputs.push_back(outputTensorAlloc);
    mlir::func::CallOp memcpyH2DCall;

    // ---------- Build argSlots / argSizes from host side ----------
    auto argSlots =
        memref::AllocOp::create(rewriter, loc,
                                MemRefType::get({(int64_t)cudaAllInputs.size()},
                                                rewriter.getI64Type()));

    auto argSizes =
        memref::AllocOp::create(rewriter, loc,
                                MemRefType::get({(int64_t)cudaAllInputs.size()},
                                                rewriter.getI64Type()));

    for (auto [i, opr] : llvm::enumerate(cudaAllInputs)) {
      auto nbytes = getNbytes(opr.getType());
      auto nbytesVal = arith::ConstantIntOp::create(rewriter, loc, nbytes, 64);
      auto device_ptr_callOp = createCallToCudaShimMalloc(
          rewriter, loc, registry, stream, nbytesVal, false);

      devicePtrs.push_back(device_ptr_callOp.getResult(0));

      auto host_ptr = getIndexFromValue(rewriter, loc, opr);

      if (i < adaptor.getOperands().size()) {
        registry.call(
            rewriter, launchGpuOp, CudaShimFn::MemcpyH2D,
            ValueRange{device_ptr_callOp.getResult(0), host_ptr, nbytesVal});
      } else {
        // this is the output tensor, we will add memcpy from device to host for
        // it after the kernel launch. and we will move this to the end of lanch
        // kernel later.
        memcpyH2DCall = registry.call(
            rewriter, launchGpuOp, CudaShimFn::MemcpyD2H,
            ValueRange{host_ptr, device_ptr_callOp.getResult(0), nbytesVal});
      }

      // constuct the argSlots and argSizes on host side for the kernel launch.
      arith::ConstantIndexOp indexVal =
          arith::ConstantIndexOp::create(rewriter, loc, i);

      memref::StoreOp::create(rewriter, loc, devicePtrs[i], argSlots,
                              ValueRange{indexVal});

      auto nElements = arith::ConstantIntOp::create(
          rewriter, loc, llvm::cast<MemRefType>(opr.getType()).getNumElements(),
          64);

      // store the size of the argument to argSizes.
      memref::StoreOp::create(rewriter, loc, nElements, argSizes,
                              ValueRange{indexVal});
    }

    // create the block size for the kernel lauch.
    auto gridAttr = launchGpuOp->getDiscardableAttr("grid");
    if (!gridAttr) {
      return rewriter.notifyMatchFailure(
          launchGpuOp, "expected 'grid' attribute to be present");
    }
    auto gridArrayAttr = llvm::dyn_cast<DenseI64ArrayAttr>(gridAttr);

    if (!gridArrayAttr || gridArrayAttr.size() != 3) {
      return rewriter.notifyMatchFailure(
          launchGpuOp,
          "expected 'grid' attribute to be an array of 3 integers");
    }

    // because of the limitation of the unsupported grid size in the cuda tile,
    // we will just use 1 for all dimensions of the grid.
    auto blockX = gridArrayAttr[0];
    auto blockY = gridArrayAttr[1];
    auto blockZ = gridArrayAttr[2];

    if (!blockX || !blockY || !blockZ) {
      return rewriter.notifyMatchFailure(
          launchGpuOp,
          "expected 'grid' attribute to be an array of 3 integers");
    }

    arith::ConstantIntOp blockXVal =
        arith::ConstantIntOp::create(rewriter, loc, blockX, 32);
    arith::ConstantIntOp blockYVal =
        arith::ConstantIntOp::create(rewriter, loc, blockY, 32);
    arith::ConstantIntOp blockZVal =
        arith::ConstantIntOp::create(rewriter, loc, blockZ, 32);

    // create the number of arguments for the kernel launch, which is the number
    // of input tensors + 1 (for the output tensor).
    auto numArgsVal =
        arith::ConstantIntOp::create(rewriter, loc, cudaAllInputs.size(), 32);

    auto argSlotPtr = getIndexFromValue(rewriter, loc, argSlots);
    auto argSizePtr = getIndexFromValue(rewriter, loc, argSizes);

    // create a call to the cuda shim function to launch the kernel.
    registry.call(rewriter, launchGpuOp, CudaShimFn::LaunchBlockPacked,
                  ValueRange{load_cubin_callee.getResult(0), kname_loaded_index,
                             blockXVal, blockYVal, blockZVal,
                             stream.getResult(0), argSlotPtr, argSizePtr,
                             numArgsVal});

    auto sync =
        registry.call(rewriter, launchGpuOp, CudaShimFn::StreamSynchronize,
                      ValueRange{stream.getResult(0)});

    memcpyH2DCall->moveAfter(sync);

    // add free after the kernel launch.
    memref::DeallocOp::create(rewriter, loc, argSlots);
    memref::DeallocOp::create(rewriter, loc, argSizes);

    for (auto operand : llvm::reverse(devicePtrs)) {
      registry.call(rewriter, launchGpuOp, CudaShimFn::Free,
                    ValueRange{operand, stream.getResult(0)});
    }

    // clean up
    registry.call(rewriter, launchGpuOp, CudaShimFn::StreamDestroy,
                  ValueRange{stream.getResult(0)});
    registry.call(rewriter, launchGpuOp, CudaShimFn::UnloadModule,
                  ValueRange{load_cubin_callee.getResult(0)});

    rewriter.replaceOp(launchGpuOp, outputTensorAlloc);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
  StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering,
               MatMulOpLowering, LanchGpuLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
