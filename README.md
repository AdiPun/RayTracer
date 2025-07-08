# RayTracer

A data-oriented, SIMD-accelerated C++ ray tracer based on *Ray Tracing in One Weekend*. This project explores high-performance techniques such as AVX2 vector batching, multithreading, and structure-of-arrays (SoA) design, with the longer-term goal of evolving into a lightweight game engine prototype.

---

## Table of Contents

- [Overview](#overview)
- [Build Instructions](#build-instructions)
- [Development Goals](#development-goals)
- [References](#references)
---

## Overview

This project reimplements the core ideas from *Ray Tracing in One Weekend* with a focus on:

- **Data-Oriented Design (DoD)**: Using SoA layouts and minimizing cache misses.
- **SIMD Acceleration**: Batched math using AVX2 intrinsics (`__m256`).
- **Multithreading (WIP)**: Preparing rendering systems for concurrency.
- **C++17 Modernization**: Clean, modular code using modern C++ features.

The intention is not only to build a fast ray tracer, but also to explore foundational systems that could feed into a future game engine, such as batching, task systems, and data-flow-driven rendering.

---

## Build Instructions
You'll need to install [cmake](https://cmake.org/download/)
```bat
# Build the project
./Tools/PungineTools/Build_Project.cmd

# Then run the raytracer
./Tools/PungineTools/Generate_Image.cmd
or
./x64/Release/RayTracer.exe > image.ppm
```
## Development Goals
- Speed: SIMD and SoA-friendly math from the ground up

- Scalability: Batch-focused and eventually multithreaded

- Simplicity: No dependencies beyond the standard library + intrinsics

- Extensibility: Open to evolving into a visualisation or engine sandbox

## References
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview)
[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
[Data-Oriented Design Book](https://www.dataorienteddesign.com/dodmain/)
[cppreference (C++17)](https://en.cppreference.com/w/cpp/17.html)

