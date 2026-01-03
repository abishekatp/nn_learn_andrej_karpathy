use ndarray::Array2;
use rand::{
    distributions::{Distribution, Uniform},
    Rng, SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// returns `([[x1,y1], [x2,y2], [x3,y3],...], [1, -1, 1,...])`
/// 1 means (x,y) belongs to upper halft moon else lower half moon
pub fn make_moons(n_samples: usize, noise: f64, seed: u64) -> (Array2<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, PI);

    // Generate points for the first moon
    let mut data = Vec::with_capacity(n_samples);
    let mut labels: Vec<f64> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let angle = uniform.sample(&mut rng);
        // the following radius is used to add some randomness to the generated values
        let radius = 1.0 + noise * rng.gen::<f64>();

        let x = radius * angle.cos();
        let y = radius * angle.sin();

        // first half circle point
        if i < n_samples / 2 {
            data.push([x, y]);
            labels.push(1.0);
        } else {
            // Shift and mirror for the second moon(half cirle points)
            data.push([x + 1.0, -y + 0.5]);
            labels.push(-1.0);
        }
    }

    (
        Array2::from_shape_vec((n_samples, 2), data.concat())
            .expect("error constructing array from vector"),
        labels,
    )
}
