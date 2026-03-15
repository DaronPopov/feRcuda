use candle_core::{D, Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.05,
            repetition_penalty: 1.1,
            seed: 42,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
        }
    }

    pub fn greedy() -> Self {
        Self::new(SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        })
    }

    pub fn sample(&mut self, logits: &Tensor, previous_tokens: &[u32]) -> CandleResult<u32> {
        let logits = logits.squeeze(0)?;
        let logits = if self.config.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
            self.apply_repetition_penalty(&logits, previous_tokens)?
        } else {
            logits
        };

        if self.config.temperature == 0.0 {
            return self.argmax(&logits);
        }

        let scaled = (&logits / self.config.temperature)?;
        let probs = candle_nn::ops::softmax(&scaled, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let filtered = if self.config.min_p > 0.0 {
            self.min_p_filter(&probs_vec)
        } else {
            probs_vec
        };
        let filtered = if self.config.top_p < 1.0 {
            self.top_p_filter(&filtered)
        } else {
            filtered
        };
        let filtered = if self.config.top_k > 0 {
            self.top_k_filter(&filtered)
        } else {
            filtered
        };

        self.sample_from_probs(&filtered)
    }

    fn argmax(&self, logits: &Tensor) -> CandleResult<u32> {
        let idx = logits.argmax(D::Minus1)?;
        Ok(idx.to_scalar::<u32>()?)
    }

    fn apply_repetition_penalty(&self, logits: &Tensor, previous_tokens: &[u32]) -> CandleResult<Tensor> {
        let penalty = self.config.repetition_penalty as f64;
        let device = logits.device();
        let vocab_size = logits.dims()[0];

        let mut token_set: Vec<u32> = previous_tokens.to_vec();
        token_set.sort_unstable();
        token_set.dedup();
        token_set.retain(|&t| (t as usize) < vocab_size);

        if token_set.is_empty() {
            return Ok(logits.clone());
        }

        let indices = Tensor::new(token_set.as_slice(), device)?;
        let selected = logits.index_select(&indices, 0)?;
        let zeros = Tensor::zeros_like(&selected)?;
        let pos_mask = selected.gt(&zeros)?;
        let divided = (&selected / penalty)?;
        let multiplied = (&selected * penalty)?;
        let penalized = pos_mask.where_cond(&divided, &multiplied)?;
        let diff = (&penalized - &selected)?;
        logits.index_add(&indices, &diff, 0)
    }

    fn top_p_filter(&self, probs: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed {
            if cumsum < self.config.top_p as f32 {
                filtered[idx] = prob;
                cumsum += prob;
            }
        }

        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }
        filtered
    }

    fn top_k_filter(&self, probs: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed.into_iter().take(self.config.top_k) {
            filtered[idx] = prob;
        }

        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }
        filtered
    }

    fn min_p_filter(&self, probs: &[f32]) -> Vec<f32> {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = self.config.min_p * max_prob;
        let mut filtered: Vec<f32> = probs
            .iter()
            .map(|&p| if p >= threshold { p } else { 0.0 })
            .collect();

        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }
        filtered
    }

    fn sample_from_probs(&mut self, probs: &[f32]) -> CandleResult<u32> {
        let r = self.random_f32();
        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return Ok(idx as u32);
            }
        }
        for (idx, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return Ok(idx as u32);
            }
        }
        Ok(0)
    }

    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }
}
