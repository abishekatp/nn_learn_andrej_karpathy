# Statistics

### Key Differences Between `WeightedIndex` and `Uniform` in Rust's `rand` Crate

Both `WeightedIndex` and `Uniform` are implementations of the `Distribution` trait in the `rand` crate (or `rand_distr` for some), used with an RNG (like `thread_rng()`) to sample random values via `.sample(&mut rng)`.

| Aspect                  | `Uniform` (from `rand_distr::Uniform` or `rand::distributions::uniform`) | `WeightedIndex` (from `rand::distributions::weighted::WeightedIndex`) |
|-------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Purpose**            | Samples **uniformly** (equal probability) from a specified range (continuous or discrete). | Samples **discretely** from a set of items with **custom weights** (non-uniform probabilities proportional to weights). |
| **Probability Distribution** | All outcomes have exactly the same chance (flat/uniform). | Outcomes have probabilities proportional to provided weights (higher weight = more likely). |
| **Typical Use Cases**   | - Random integers in a range: `Uniform::new(0, 10)` for 0..10.<br>- Fair dice rolls, random floats in [low, high).<br>- Basic uniform randomness. | - Weighted random choice: e.g., select index from `[10, 1, 5]` → item 0 ~71% chance, item 1 ~7%, item 2 ~36%.<br>- Bigram/char sampling in language models (like your name generator).<br>- Loot tables in games. |
| **Creation**           | `Uniform::new(low, high)` or `new_inclusive`.<br>Panics if low >= high. | `WeightedIndex::new(&weights_vec)` where weights are `usize`, `u32`, etc.<br>Errors if all weights zero or overflow. |
| **Sampling Output**    | A value directly from the range (e.g., `i32`, `f64`). | An **index** ( `usize` ) into the original weights vector. |
| **Internal Mechanism** | Direct uniform RNG mapping to range. | Uses alias method or similar for O(1) sampling after O(n) prep; internally relies on a `Uniform` for randomness. |
| **Performance**        | Very fast (direct). | Fast after construction (O(n) setup, O(1) sample). |
| **Dependencies**       | Core in `rand_distr`. | In `rand` crate's weighted module. |

### Quick Code Examples

**Uniform** (equal chance for each integer):
```rust
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

let mut rng = thread_rng();
let die = Uniform::new_inclusive(1, 6);  // Fair 6-sided die
let roll = die.sample(&mut rng);
println!("Rolled: {}", roll);  // Each 1-6 equally likely
```

**WeightedIndex** (non-uniform, like your bigram probs):
```rust
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

let mut rng = thread_rng();
let weights = [5, 1, 3];  // Higher weight = more likely
let dist = WeightedIndex::new(&weights).unwrap();
let index = dist.sample(&mut rng);  // e.g., 0 ~56%, 1 ~11%, 2 ~33%
println!("Selected index: {}", index);
```

In your bigram name generator, `WeightedIndex` is perfect because your row in the probability matrix has varying "weights" (probabilities), while `Uniform` would pick characters equally (ignoring learned frequencies).

`WeightedIndex` is essentially a **weighted discrete** sampler, while `Uniform` is the base for **equal-probability** sampling (and even used internally by `WeightedIndex`). For statistical prediction from counts/probs, stick with `WeightedIndex`!