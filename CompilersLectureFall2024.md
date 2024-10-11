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
            ```

2. **Reopen in Container:**
    - Press `F1` to open the command palette or press `Cmd+Shift+P` on macOS
    - Type `Dev Containers: Rebuild and Reopen in Container` and select it.
    - VS Code will now build and start the devcontainer.

## Building LLVM and MLIR

Once you have the devcontainer set up, you can build LLVM and MLIR using the provided script.

1. **Open the Integrated Terminal:**
    - Press `` Ctrl+` `` to open the integrated terminal in VS Code.

2. **Run the Build Script:**

```bash
./build_tools/build_llvm_dev.sh \
  $PWD/external/llvm-project/ \
  $PWD/builds/llvm-project/build/ \
  $PWD/builds/llvm-project/install/
```

## Exploring MLIR

Here are some important directories and files related to MLIR:

- [MLIR Library](mlir/lib)
- [MLIR Include Files](mlir/include)
- [MLIR Tests](mlir/test)

## Additional Resources

- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/docs/)

## Templates for Other Sections

### Introduction to MLIR

Overview of MLIR, its purpose, and its architecture.

### Basic MLIR Operations

Basic operations in MLIR, including how to create and manipulate MLIR files.

### Advanced MLIR Topics

Advanced topics such as custom dialects, passes, and optimizations.

### Assignments and Exercises

Assignments and exercises for students to practice MLIR concepts.
