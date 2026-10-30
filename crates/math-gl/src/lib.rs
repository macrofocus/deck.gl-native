//! Rust port of the parts of [math.gl](https://math.gl) that deck.gl-native needs.
//!
//! Module layout mirrors the JavaScript monorepo: `web_mercator` corresponds to
//! `@math.gl/web-mercator` and `culling` to `@math.gl/culling`. Vector and matrix types come
//! from `glam` in f64
//! precision, which matches the JavaScript implementation's use of doubles.

pub mod culling;
pub mod fly_to;
pub mod web_mercator;

pub use glam;
