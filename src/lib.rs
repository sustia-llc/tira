use nalgebra::{DMatrix, DVector, Matrix, Matrix3};
use rand::prelude::*;
use rand::seq::WeightError;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Bernoulli, BernoulliError, Distribution};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use thiserror::Error;
mod simulation;
mod plotter;
mod agent;

#[derive(Error, Debug)]
pub enum OneManyError {
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),
    #[error("Invalid action: {0}")]
    InvalidAction(usize),
    #[error("Distribution error: {0}")]
    Distribution(#[from] BernoulliError),
    #[error("Weight error: {0}")]
    Weight(#[from] WeightError),
}

// Environment trait for different bandit environments
pub trait Environment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError>;
}

// Basic multi-armed bandit environment
#[derive(Debug, Clone)]
pub struct BanditEnvironment {
    probabilities: Vec<f64>,
    rng: ThreadRng,
}

impl BanditEnvironment {
    pub fn new(probabilities: Vec<f64>) -> Result<Self, OneManyError> {
        // Validate probabilities are between 0 and 1
        for p in &probabilities {
            if *p < 0.0 || *p > 1.0 {
                return Err(OneManyError::InvalidProbability(*p));
            }
        }

        Ok(Self {
            probabilities,
            rng: rand::rng(),
        })
    }
}

impl Environment for BanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError> {
        if action >= self.probabilities.len() {
            return Err(OneManyError::InvalidAction(action));
        }

        let prob = self.probabilities[action];
        let dist = Bernoulli::new(prob).map_err(OneManyError::Distribution)?;

        Ok(if dist.sample(&mut self.rng) { 1 } else { 0 })
    }
}

