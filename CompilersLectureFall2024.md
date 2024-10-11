# Compilers Class - MLIR

Welcome to the Compilers Class on MLIR! This guide will help you set up the LLVM project using devcontainers in Visual Studio Code, so you can quickly start compiling the project and playing with the MLIR tools.

## Prerequisites

To get started, you need to have the following installed:

- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker](https://www.docker.com/)

## Setting Up Devcontainers

1. **Open the Project in Visual Studio Code:**
    - Open Visual Studio Code.
    - Open the folder containing this project or:
        - Shallow clone the project at the branch you want to work on:
            ```sh
            git clone --depth 1 --branch 19.1.1-20241011-neu_class https://github.com/agostini01/llvm-project.git
            cd llvm-project
            code .
            ```

2. **Reopen in Container:**
    - Open the command palette with `Ctrl+Shift+P` or `Cmd+Shift+P`
    - Type `Dev Containers: Rebuild and Reopen in Container` and select it.
    - VS Code will now build and start the devcontainer.

## Building LLVM and MLIR

Once you have the devcontainer set up, you can build LLVM and MLIR using the provided script.

1. **Open the Integrated Terminal:**
    - Press `` Ctrl+` `` to open the integrated terminal in VS Code select a new bash shell.

2. **Run the Build Script:**

First time:

```bash
./build_tools/build_llvm_dev.sh \
  $PWD/ \
  $PWD/builds/llvm-project/build/ \
  $PWD/builds/llvm-project/install/
```

Subsequent builds:


```bash
# cmake --build /workspaces/llvm-project/builds/llvm-project/build/ --target <desired_target_list>
cmake --build /workspaces/llvm-project/builds/llvm-project/build/ --target check-mlir # compile and run tests
cmake --build /workspaces/llvm-project/builds/llvm-project/build/ --target opt mlir-opt mlir-translate mlir-cpu-runner install # install 
```


## Examples used in the class

I have included a convenient "Run test on file" vscode task that you can use to
run llvm-project testing on a specific test file.  This can be achieved by:

1. Open the file you want to test inside the `mlir/test` directory.
2. Open the command palette with `Ctrl+Shift+P` or `Cmd+Shift+P`
3. Type `Tasks: Run Task` and select it.
4. Select `Run llvm-lit on file` from the list of tasks.

Some examples used in the class are:

- [Common Subexpression Elimination](mlir/test/CompilersLectureFall2024/cse.mlir)
- [Lower affine](mlir/test/CompilersLectureFall2024/lower-affine.mlir)
- [Lower linalg](mlir/test/CompilersLectureFall2024/lower-linalg.mlir)
- [Translate ll into MLIR](mlir/test/CompilersLectureFall2024/mlir-translate.ll)
- [Translate mlir llvm dialect into ll](mlir/test/CompilersLectureFall2024/mlir-translate.mlir)
- [Schedule transformations with the transform dialect](mlir/test/CompilersLectureFall2024/transform-e2e.mlir)

## Exploring MLIR Source Code

Here are some important directories and files related to MLIR:

- [MLIR Include Files and Dialects](mlir/include/mlir/Dialect/)
- [MLIR Generic Custom Transformations](mlir/lib/Transforms/)
- [MLIR Tests](mlir/test)


## Debugging MLIR

I have included a convenient `launch.json` configuration that you can use to
debug the MLIR tools.  Let's set a breakpoint in one of the passes.

1. **Open the File:**
    - Navigate to [`mlir/lib/Dialect/Affine/Transforms/LoopUnroll.cpp`](mlir/lib/Dialect/Affine/Transforms/LoopUnroll.cpp#L93).
2. **Set a Breakpoint:**
    - Scroll to line 93.
    - Click on the left margin next to the line number to set a breakpoint.
3. **Start Debugging:**
    - Open the command palette with `Ctrl+Shift+P` or `Cmd+Shift+P`.
    - Type `Debug: Select and Start Debugging` and select it.
    - Choose the appropriate debugging configuration from the list.

4. **Inspect Variables and Step Through Code:**
    - Use the debugging toolbar to step through the code, inspect variables, and evaluate expressions.

## Additional Resources

- Docs: https://mlir.llvm.org/
- Source Code: https://github.com/llvm/llvm-project/tree/main/mlir
- Forums: https://discourse.llvm.org/c/mlir/31
