//! Open-loop sinusoidal per-actuator controller.
//!
//! Genome:
//! * `freq_hz` — global oscillation frequency in `[0.5, 4]` Hz.
//! * `phases[i]` — per-actuator phase in `[0, 2π]`.
//! * `amps[i]` — per-actuator amplitude (fraction of base edge length) in
//!   `[0.05, 0.30]`.
//!
//! Output: at time `t` and actuator `i`, the rest-length offset is
//! `ΔL = base_edge_len · amp[i] · sin(2π · freq_hz · t + phases[i])`. This
//! offset is added to every edge spring the actuator modulates (see
//! `physics::World::substep` for how multi-actuator contributions are summed).

use rand::{Rng, RngCore};

/// Gene bounds — match the spec.
pub const FREQ_MIN: f32 = 0.5;
pub const FREQ_MAX: f32 = 4.0;
pub const AMP_MIN: f32 = 0.05;
pub const AMP_MAX: f32 = 0.30;

#[derive(Debug, Clone)]
pub struct Controller {
    pub freq_hz: f32,
    pub phases: Vec<f32>,
    pub amps: Vec<f32>,
}

impl Controller {
    /// Random controller with `n_actuators` actuators, freq ∈ [FREQ_MIN..FREQ_MAX].
    pub fn random(rng: &mut (impl RngCore + ?Sized), n_actuators: usize) -> Self {
        let freq_hz = rng.random_range(FREQ_MIN..=FREQ_MAX);
        let mut phases = Vec::with_capacity(n_actuators);
        let mut amps = Vec::with_capacity(n_actuators);
        for _ in 0..n_actuators {
            phases.push(rng.random_range(0.0..std::f32::consts::TAU));
            amps.push(rng.random_range(AMP_MIN..=AMP_MAX));
        }
        Self {
            freq_hz,
            phases,
            amps,
        }
    }

    /// All-zeros controller: every actuator fully contracted (ΔL = 0). Useful
    /// for the morph-only baseline.
    pub fn zeros(n_actuators: usize) -> Self {
        Self {
            freq_hz: 1.0,
            phases: vec![0.0; n_actuators],
            amps: vec![0.0; n_actuators],
        }
    }

    /// Resize in place to `new_n` actuators, sampling new entries at random
    /// from the uniform ranges — used when morphology mutation grows the
    /// actuator count.
    pub fn resize(&mut self, new_n: usize, rng: &mut (impl RngCore + ?Sized)) {
        self.phases.resize(new_n, 0.0);
        self.amps.resize(new_n, 0.0);
        for i in 0..new_n {
            if i >= self.phases.len().saturating_sub(new_n)
                && self.phases[i] == 0.0
                && self.amps[i] == 0.0
            {
                self.phases[i] = rng.random_range(0.0..std::f32::consts::TAU);
                self.amps[i] = rng.random_range(AMP_MIN..=AMP_MAX);
            }
        }
        // Always top up new entries (the above heuristic is unreliable).
        for i in 0..new_n {
            if self.amps[i] == 0.0 {
                self.phases[i] = rng.random_range(0.0..std::f32::consts::TAU);
                self.amps[i] = rng.random_range(AMP_MIN..=AMP_MAX);
            }
        }
    }

    /// Compute the rest-length offset for actuator `i` at world-time `t`.
    #[inline]
    pub fn delta_l(&self, i: usize, t: f32) -> f32 {
        let amp = *self.amps.get(i).unwrap_or(&0.0);
        let phase = *self.phases.get(i).unwrap_or(&0.0);
        amp * (std::f32::consts::TAU * self.freq_hz * t + phase).sin()
    }

    /// Mutate the controller with gaussian perturbation, clipped to valid
    /// ranges. `sigma_freq` in Hz, `sigma_phase` in radians, `sigma_amp`
    /// fraction units.
    pub fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, p: MutationParams) {
        self.freq_hz += rng.random::<f32>() * 2.0 * p.sigma_freq - p.sigma_freq;
        self.freq_hz = self.freq_hz.clamp(FREQ_MIN, FREQ_MAX);
        for phase in self.phases.iter_mut() {
            *phase += rng.random::<f32>() * 2.0 * p.sigma_phase - p.sigma_phase;
            *phase = phase.rem_euclid(std::f32::consts::TAU);
        }
        for amp in self.amps.iter_mut() {
            *amp += rng.random::<f32>() * 2.0 * p.sigma_amp - p.sigma_amp;
            *amp = amp.clamp(0.0, AMP_MAX);
            if *amp < AMP_MIN {
                *amp = AMP_MIN;
            } // re-floored
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MutationParams {
    pub sigma_freq: f32,
    pub sigma_phase: f32,
    pub sigma_amp: f32,
}

impl Default for MutationParams {
    fn default() -> Self {
        Self {
            sigma_freq: 0.2,
            sigma_phase: 0.4,
            sigma_amp: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn random_controller_respects_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..50 {
            let c = Controller::random(&mut rng, 5);
            assert!(c.freq_hz >= FREQ_MIN && c.freq_hz <= FREQ_MAX);
            assert_eq!(c.phases.len(), 5);
            assert_eq!(c.amps.len(), 5);
            for &p in &c.phases {
                assert!((0.0..=std::f32::consts::TAU).contains(&p));
            }
            for &a in &c.amps {
                assert!((AMP_MIN..=AMP_MAX).contains(&a));
            }
        }
    }

    #[test]
    fn delta_l_is_bounded() {
        let c = Controller {
            freq_hz: 2.0,
            phases: vec![0.0],
            amps: vec![0.3],
        };
        let mut max_abs = 0.0;
        for k in 0..1000 {
            let t = k as f32 * 0.001;
            let d = c.delta_l(0, t);
            assert!(d.abs() <= 0.3001, "delta_l out of amp range: {d}");
            if d.abs() > max_abs {
                max_abs = d.abs();
            }
        }
        assert!(max_abs > 0.25, "amplitude never reached near max");
    }

    #[test]
    fn mutate_keeps_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut c = Controller::random(&mut rng, 4);
        for _ in 0..200 {
            c.mutate(&mut rng, MutationParams::default());
            assert!(c.freq_hz >= FREQ_MIN && c.freq_hz <= FREQ_MAX);
            for &a in &c.amps {
                assert!((AMP_MIN - 1e-4..=AMP_MAX + 1e-4).contains(&a));
            }
            for &p in &c.phases {
                assert!((0.0..=std::f32::consts::TAU + 1e-3).contains(&p));
            }
        }
    }
}
