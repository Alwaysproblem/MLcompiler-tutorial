include "mlir/Pass/PassBase.td"

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
