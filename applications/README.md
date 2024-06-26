# HoloHub Applications

This directory contains applications based on the Holoscan Platform.
Some applications might require specific hardware and software packages which are described in the 
metadata.json and/or README.md for each application.

# Contributing to HoloHub Applications

Please review the [CONTRIBUTING.md file](https://github.com/nvidia-holoscan/holohub/blob/main/CONTRIBUTING.md) guidelines to contribute applications.

# HoloHub Application Organization Conventions

## Required Conventions

We expect that an application contributed to HoloHub conforms to the following organization:

- Each project must provide a `metadata.json` file reflecting several key components such as the application name and description, authors, dependencies, and the primary project language. 
- Each project must provide a `README` or `README.md` file.
  - We strongly recommend that the project `README` file provides at least the information given in the [template README](./template/README.md.template), including the project description and a splash image.
- Each project must be organized in its own subfolder under `holohub/applications/`.

See the [HoloHub application template](./template/) for example `README` and `metadata.json` documents to get started.

## Recommended Conventions

Contributors may additionally opt to lay out their project structure in a way that conforms to HoloHub conventions in order to enable common infrastructure for their project, including streamlined build and run support in the [`dev_container`](../dev_container) and [`run`](../run) scripts and search support on the HoloHub landing page.

If your code does not adhere to these conventions, please set the field `manual_setup` to `true` in your project `metadata.json` file to opt out and indicate that your project is not eligible for streamlined infrastructure support.

HoloHub recommended application convention is as follows:
- Languages
  - Project is either C++ or Python language
  - If multiple language implementations are provided, each must be added to its own language subfolder as follows:
```
applications/
  └── my_project/
        ├── cpp/
        │     ├── ...
        │     └── ...
        └── python/
              ├── ...
              └── ...
```

- Container Environment
  - Project may provide its own container environment or opt to use the default HoloHub environment
  - If the project specifies its own container:
    - Default project environment must be named `Dockerfile`
    - Project `Dockerfile` must be located at either:
      - The same directory as `metadata.json`, or
      - If the project defines a language directory (`cpp`, `python`), may provide at the project folder one level above, or
      - The project may provide an alternative default Dockerfile path in `metadata.json`
  - If the project does not specify a `Dockerfile` then the [default HoloHub `Dockerfile`](../Dockerfile) will be used

- Build and Run Instructions
  - Must provide a run command in `metadata.json` for `./run launch` to reference
  - Must otherwise comply with `./dev_container build_and_run` command options for the default use case
  - Advanced instructions may also be specified in the project README
