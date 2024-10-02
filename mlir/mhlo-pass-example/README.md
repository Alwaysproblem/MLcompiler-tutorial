# MLIR Pass Tutor

In this tutorial, we will walk through the process of writing a simple pass for the MLIR infrastructure. We will start with a simple pass and then we will add functionality to it step by step. The tutorial will take [`mhlo`](https://github.com/tensorflow/mlir-hlo) as an basic dialect instead of a new dialect. The reason is that `mhlo` is a dialect that is already implemented and we can use it as a reference.

There are several pass implemented in this tutorial:

- Pow2 pass: $x^2$ -> $x\times{x}$
- Tanh pass: tanh(x) -> $\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
- ExpLog pass: $e^{ln(a) + ln(b)}$ -> $a\times{b}$
- Inline pass: inline function call (just for demonstration, please use the `mlir::createInlinerPass()` function in your project instead.)
- Outline pass: pack a block of operations into a function and insert the `func::Callop` in the main region.
- LoopFusion pass: fuse two loops into one loop (TODO)

For the [`pass`](https://mlir.llvm.org/docs/PassManagement/) declaration, there are 2 ways to do it:

- Use [`tddr`](https://mlir.llvm.org/docs/DeclarativeRewrites) to generation
- Write cpp code by hand

For the `pass` implementation, there are 3 ways to do it:

- Write cpp code by hand (out of scope of this tutorial)
- Use [`tddr`](https://mlir.llvm.org/docs/DeclarativeRewrites) to generation
- Use [`pdl`](https://mlir.llvm.org/docs/PDLL/) language to generation

The Pow2 part will illustrate those ways to implement a pass.

## Environment Setup

### Environment Preparation with dev containers

Please choose the `Dev Containers: Open Folder in Container...`

- build example with dev containers

```bash
cd mhlo-phlo-prototype
bash build_tools/sync_deps.sh
bash build_tools/build_deps.sh
bash build.sh check-mhlo-pass-tutor
```

## Configure the Clangd

```bash
cd mhlo-phlo-prototype
# after you configure the project with cmake, you can configure the clangd by run the following command
compdb -p build list > compile_commands.json
```

## Pow 2 into x by x substitution pass

Here is the example for pow2 pass:

- Before pow2 pass:

```mlir
module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = mhlo.constant dense<2.0> : tensor<2x2xf32>
    %2 = "mhlo.power"(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %2 : tensor<2x2xf32>
  }
}
```

- After pow2 pass:

```mlir
module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "mhlo.multiply"(%0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %2 : tensor<2x2xf32>
  }
}
```

### Table-driven Declarative Rewrite Rule

In addition to subclassing the `mlir::RewritePattern` C++ class, MLIR also supports defining rewrite rules in a declarative manner. Similar to Op Definition Specification (ODS), this is achieved via TableGen, which is a language to maintain records of domain-specific information. The rewrite rules are specified concisely in a TableGen record, which will be expanded into an equivalent `mlir::RewritePattern` subclass at compiler build time.

This is to say, we can use the `td` file to generate the cpp code and avoid some duplicatd work !!

Take the pow2 pass as an example:

we can define the `td` file like this:

```td
def Pow2OptPattern : Pat<(MHLO_PowOp $arg, (MHLO_ConstantOp:$cst $cstVal)),
                         (MHLO_MulOp $arg, $arg),
                         [(TypesAreIdentical $arg, $cst),
                          (Eqn2 $cst)]>;
```

`Pat` is the alias for `Pattern` when the destination graph is a sinlgle node.

```cpp
// Note: The DRR definition used for defining patterns is shown below:
class Pattern<
   dag sourcePattern, list<dag> resultPatterns,
   list<dag> additionalConstraints = [],
   dag benefitsAdded = (addBenefit 0)
>;
```

The `sourcePatten` of the Pattern in the Pow2OptPattern is:

```td
(MHLO_PowOp $arg, (MHLO_ConstantOp:$cst $cstVal))
```

This pattern will match the DAG graph:

```mlir
%1 = mhlo.constant dense<2.0> : tensor<2x2xf32>
%2 = "mhlo.power"(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
```

The `resultPatterns` in the Pow2OptPattern is:

```mlir
%2 = "mhlo.multiply"(%0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
```

The `additionalConstraints` in the Pow2OptPattern is:

```td
[(TypesAreIdentical $arg, $cst),
 (Eqn2 $cst)]
```

The `Eqn2` Constraint is defined with `def Eqn2 : Constraint<CPred<"::ValueEql2($0)">>;`

the CPred is a condition predicate, which is defined with `def CPred<str s> : Constraint<CPred<"C++ Code">>;`, the strings in the `CPred` is the C++ code which means the C++ helper function. you can find the `ValueEql2` in the `Pow2/mlir/Pow2.cpp:26` file. You can also define your own C++ Code instead of using the helper function. At the same time, the `NativeCodeCall` also can use the C++ code.

The `Eqn2` checks the `cst` is equal to 2.

The generated cpp code is:

```cpp
struct Pow2OptPattern : public ::mlir::RewritePattern {
  Pow2OptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.power", 2, context, {"mhlo.multiply"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::ElementsAttr cstVal;
    ::mlir::mhlo::ConstantOp cst;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::PowOp>(op0); (void)castedOp0;
    arg = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::mhlo::ConstantOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::mhlo::ConstantOp type";
        });
      }
      cst = castedOp1;
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ElementsAttr>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'mhlo.constant' to have attribute 'value' of type '::mlir::ElementsAttr'";
          });
        }
        cstVal = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    if (!(((*arg.begin()).getType() == (*cst.getODSResults(0).begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'arg, cst' failed to satisfy constraint: ''";
      });
    }
    if (!((::ValueEql2((*cst.getODSResults(0).begin()))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'cst' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::MulOp tblgen_MulOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      tblgen_values.push_back((*arg.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_0 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<Pow2OptPattern>(patterns.getContext());
}
```

You can find more detail in [TDDR](https://mlir.llvm.org/docs/DeclarativeRewrites/)

### PDL Language

Pattern matching is an extremely important component within MLIR, as it encompasses many different facets of the compiler. From canonicalization, to optimization, to conversion; every MLIR based compiler will heavily rely on the pattern matching infrastructure in some capacity.

The PDL Language (PDLL) provides a declarative pattern language designed from the ground up for representing MLIR pattern rewrites. PDLL is designed to natively support writing matchers on all of MLIRs constructs via an intuitive interface that may be used for both ahead-of-time (AOT) and just-in-time (JIT) pattern compilation.

The PDLL file will generate the MLIR PDLL Dialect format and embed it into a cpp string in the generative C++ code.

Compared with `td` file, the `pdll` file is more like a high-level language, which is more readable and easier to write. However, `pdll` is not as flexible as td file, and it is not easy to debug the problem.

Take the pow2 pass as an example:

````pdll
// this is Constaint function.
Constraint Eqn2(value: Value);

Constraint TypesAreIdentical(value1: Value, value2: Value)[{
  return success(value1.getType() == value2.getType());
}];

Pattern Pow2PdllOptPattern with benefit(0) {
  // ** match section ** //
  // This is declare for a Value. and `: Value` is also a constraint.
  let const_2 : Value = op<mhlo.constant>();
  let arg : Value;
  TypesAreIdentical(arg, const_2);
  // This is also a constraint, which is defined in the `Pow2/mlir/Pow2.cpp:26` file.
  // This is also can be defined by :
  //
  // ```pdll
  // Constraint Eqn2(value: Value) [{
  //   // here, assuming that you already implement the ValueEql2 function.
  //   return success(ValueEql2(value));
  // }]
  // ```
  //
  Eqn2(const_2);
  let root = op<mhlo.power>(arg, const_2);

  // ** rewrite section ** //
  replace root with op<mhlo.multiply>(arg, arg);
}

````

The generated cpp code is like:

```cpp
static ::mlir::LogicalResult TypesAreIdenticalPDLFn(::mlir::PatternRewriter &rewriter, ::mlir::Value value1, ::mlir::Value value2) {
  return success(value1.getType() == value2.getType());
}

namespace {

struct Pow2PdllOptPattern : ::mlir::PDLPatternModule {
  template <typename... ConfigsT>
  Pow2PdllOptPattern(::mlir::MLIRContext *context, ConfigsT &&...configs)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
R"mlir(pdl.pattern @Pow2PdllOptPattern : benefit(0) {
  %0 = types loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":24:25)
  %1 = operation "mhlo.constant"  -> (%0 : !pdl.range<type>) loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":24:25)
  %2 = result 0 of %1 loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":24:25)
  %3 = operand loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":25:7)
  apply_native_constraint "TypesAreIdentical"(%3, %2 : !pdl.value, !pdl.value) loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":26:3)
  apply_native_constraint "Eqn2"(%2 : !pdl.value) loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":27:3)
  %4 = types loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":28:14)
  %5 = operation "mhlo.power"(%3, %2 : !pdl.value, !pdl.value)  -> (%4 : !pdl.range<type>) loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":28:14)
  rewrite %5 {
    %6 = operation "mhlo.multiply"(%3, %3 : !pdl.value, !pdl.value)  loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":29:21)
    replace %5 with %6 loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":29:3)
  } loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":29:3)
} loc("/root/Desktop/dockerVolumn/MLcompiler-tutorial/mlir/mhlo-pass-example/Pow2/mlir/Pow2.pdll":22:1)
    )mlir", context), std::forward<ConfigsT>(configs)...) {
    registerConstraintFunction("TypesAreIdentical", TypesAreIdenticalPDLFn);
  }
};

} // end namespace

template <typename... ConfigsT>
static void LLVM_ATTRIBUTE_UNUSED populateGeneratedPDLLPatterns(::mlir::RewritePatternSet &patterns, ConfigsT &&...configs) {
  patterns.add<Pow2PdllOptPattern>(patterns.getContext(), configs...);
}

```

You can find more detail in [PDLL](https://mlir.llvm.org/docs/PDLL/)

### Pass Declaration

The `td` File also can generate the pass declaration code. The `td` file will generate the `impl::Pow2PassBase` class, which can be the base class of the custom class (like `SubstitutePow2PdllGenPass` in the file `Pow2/mlir/Pow2.cpp:95`).

```td
def Pow2Pass : Pass<"pow-2", "func::FuncOp"> {
  let summary = "Pow 2 Substitution";
  let description = [{
    Here we rewrite x^2 => x * x
  }];

  // A constructor must be provided to specify how to create a default instance
  // of Pow2PassBase. It can be skipped for this specific example, because both the
  // constructor and the registration methods live in the same namespace.
  let constructor = "mhlo::createSubstitutePow2Pass()";

  // Specify any options.
  let options = [
    Option</*C++ variable name=*/"Pow2Pass", /*argument*/"pow2-opt",
           /*type*/"bool", /*default=*/"false",
           /*description*/"Enable Pow2 Optimization">,
  ];

  // Specify any statistics.
  let statistics = [
    // Statistic</*C++ variable name=*/"statistic",
    //           /*display name*/"example-statistic",
    //           /*description*/"An example statistic">
  ];
}
```

The generated cpp code is:

```cpp
...
namespace impl {

template <typename DerivedT>
class Pow2PassBase : public ::mlir::OperationPass<func::FuncOp> {
public:
  using Base = Pow2PassBase;

  Pow2PassBase() : ::mlir::OperationPass<func::FuncOp>(::mlir::TypeID::get<DerivedT>()) {}
  Pow2PassBase(const Pow2PassBase &other) : ::mlir::OperationPass<func::FuncOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("pow-2");
  }
  ::llvm::StringRef getArgument() const override { return "pow-2"; }

  ::llvm::StringRef getDescription() const override { return "Pow 2 Substitution"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("Pow2Pass");
  }
  ::llvm::StringRef getName() const override { return "Pow2Pass"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Pow2PassBase<DerivedT>)

protected:
private:
};
} // namespace impl
...
```

You can write an Custom Pass class like this:

```cpp
namespace {
struct SubstitutePow2PdllGenPass
    : impl::Pow2PassBase<SubstitutePow2PdllGenPass> {
  void runOnOperation() final {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    // --- insert the native constraints ---
    registerNativeConstraints(patterns);
    // --- insert the native constraints ---
    patterns.add<Pow2PdllOptPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace
```

And then, you can also find the `registerPasses` in the `Pow2Pass.inc` file. This can be used to register the pass to the MLIR PassManager when you want to register the pass into the `xxx-opt` binary.

```cpp
//===----------------------------------------------------------------------===//
// Pow2Pass Registration
//===----------------------------------------------------------------------===//

inline void registerPow2Pass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mhlo::createSubstitutePow2Pass();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerPow2PassPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mhlo::createSubstitutePow2Pass();
  });
}

//===----------------------------------------------------------------------===//
//  Registration
//===----------------------------------------------------------------------===//

inline void registerPasses() {
  registerPow2Pass();
}
```

You can find the example in the `pass-tutor-opt/pass-tutor-opt.cpp` file.

```cpp
  // Register all "core" dialects
  ...
  mlir::mhlo::registerPasses();
  ...
  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "pass-tutor-opt", registry).succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
```
