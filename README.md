# <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Telegram-Animated-Emojis/main/Objects/Magnifying%20Glass%20Tilted%20Left.webp" alt="Magnifying Glass Tilted Left" width="30" height="30" /> Rust ML from Scratch

Classic machine learning algorithms implemented from scratch in Rust, with no ML libraries. Each algorithm lives in its own folder (a Cargo workspace member), so the repo grows one algorithm at a time.



## 🦀 Algorithms

| Folder | Algorithm | Status |
|---|---|---|
| `linear_regression/` | Linear regression with gradient descent | In progress |


## 🦀 Project structure

```
rust-ml-from-scratch/
├── Cargo.toml              # workspace root
├── linear_regression/
│   ├── Cargo.toml
│   └── src/main.rs
└── ...                     # one folder per algorithm
```

## 🦀 Why Rust?

Rust makes data ownership and memory explicit, so writing algorithms by hand also teaches how the data actually moves. This repo is also my way of learning Rust through real problems.

I also want to experiment with using Rust for parts of my ML pipelines where performance matters, such as data processing and inference. The goal is to understand where Rust can provide fast, efficient components alongside the rest of an ML workflow.

## 🦀 Goals

- Implement classic ML algorithms without ML libraries
- Understand the mathematics and mechanics behind each algorithm
- Learn Rust through practical ML problems
- Experiment with Rust for performance-sensitive parts of ML pipelines
