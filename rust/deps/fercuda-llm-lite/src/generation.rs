use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.05,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            seed: 42,
        }
    }
}

impl GenerationConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            ..Default::default()
        }
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[derive(Debug, Clone)]
pub enum StopCondition {
    Eos(u32),
    TokenIds(Vec<u32>),
    Text(String),
    MaxTokens(usize),
}

impl StopCondition {
    pub fn should_stop(&self, token_id: u32, generated_text: &str, token_count: usize) -> bool {
        match self {
            StopCondition::Eos(eos) => token_id == *eos,
            StopCondition::TokenIds(ids) => ids.contains(&token_id),
            StopCondition::Text(seq) => generated_text.ends_with(seq),
            StopCondition::MaxTokens(max) => token_count >= *max,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    start_time: Option<Instant>,
    decode_start: Option<Instant>,
}

impl GenerationStats {
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            generated_tokens: 0,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            start_time: None,
            decode_start: None,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn end_prefill(&mut self) {
        if let Some(start) = self.start_time {
            self.prefill_time = start.elapsed();
            self.decode_start = Some(Instant::now());
        }
    }

    pub fn record_token(&mut self) {
        self.generated_tokens += 1;
        if let Some(decode_start) = self.decode_start {
            self.decode_time = decode_start.elapsed();
        }
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_time.as_secs_f64() > 0.0 {
            self.generated_tokens as f64 / self.decode_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn time_to_first_token(&self) -> Duration {
        self.prefill_time
    }

    pub fn total_time(&self) -> Duration {
        self.prefill_time + self.decode_time
    }
}

pub trait InferenceModel {
    fn forward(
        &mut self,
        tokens: &[u32],
        pos: usize,
    ) -> Result<candle_core::Tensor, Box<dyn std::error::Error>>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<candle_core::Tensor, Box<dyn std::error::Error>>;
    fn clear_cache(&mut self) {}
}

pub struct StreamingInference<'a, M: InferenceModel> {
    model: &'a mut M,
    sampler: crate::Sampler,
    config: GenerationConfig,
    stats: GenerationStats,
    decoder: crate::tokenizer::StreamDecoder,
    pos: usize,
    next_token: Option<u32>,
    eos_token: u32,
    all_tokens: Vec<u32>,
    finished: bool,
    context_length: Option<usize>,
}

impl<'a, M: InferenceModel> StreamingInference<'a, M> {
    pub fn new(
        model: &'a mut M,
        tokenizer: &'a crate::Tokenizer,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoding = tokenizer.encode(prompt)?;
        let tokens = encoding.ids;
        let eos_token = tokenizer.eos_token_id().unwrap_or(2);

        let mut stats = GenerationStats::new(tokens.len());
        stats.start();
        let logits = model.prefill(&tokens)?;
        stats.end_prefill();

        let mut sampler = crate::Sampler::new(crate::SamplerConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        });
        let next_token = sampler.sample(&logits, &tokens)?;
        let decoder = tokenizer.decode_stream(&tokens, true);
        let pos = tokens.len();

        Ok(Self {
            model,
            sampler,
            config,
            stats,
            decoder,
            pos,
            next_token: Some(next_token),
            eos_token,
            all_tokens: tokens,
            finished: false,
            context_length: None,
        })
    }

    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    fn slide_window(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let keep_count = (self.all_tokens.len() as f32 * 0.8) as usize;
        let drop_count = self.all_tokens.len() - keep_count;
        self.all_tokens = self.all_tokens.split_off(drop_count);
        self.model.clear_cache();
        let _ = self.model.prefill(&self.all_tokens)?;
        self.pos = self.all_tokens.len();
        Ok(())
    }

    pub fn next(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.finished {
            return Ok(None);
        }

        let current_token = match self.next_token {
            Some(t) => t,
            None => return Ok(None),
        };

        if current_token == self.eos_token || self.stats.generated_tokens >= self.config.max_tokens {
            self.finished = true;
            return Ok(self.decoder.flush()?);
        }

        self.all_tokens.push(current_token);
        self.stats.record_token();

        if let Some(max_ctx) = self.context_length {
            if self.pos >= max_ctx.saturating_sub(1) {
                self.slide_window()?;
            }
        }

        let text = self.decoder.step(current_token)?;
        let logits = self.model.forward(&[current_token], self.pos)?;
        self.pos += 1;
        self.next_token = Some(self.sampler.sample(&logits, &self.all_tokens)?);
        Ok(text.or(Some(String::new())))
    }

    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

pub struct SpeculativeInference<'a, T: InferenceModel, D: InferenceModel> {
    target: &'a mut T,
    draft: &'a mut D,
    sampler: crate::Sampler,
    config: GenerationConfig,
    stats: GenerationStats,
    decoder: crate::tokenizer::StreamDecoder,
    pos: usize,
    next_token: Option<u32>,
    eos_token: u32,
    all_tokens: Vec<u32>,
    finished: bool,
    draft_k: usize,
    context_length: Option<usize>,
}

impl<'a, T: InferenceModel, D: InferenceModel> SpeculativeInference<'a, T, D> {
    pub fn new(
        target: &'a mut T,
        draft: &'a mut D,
        tokenizer: &'a crate::Tokenizer,
        prompt: &str,
        config: GenerationConfig,
        draft_k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoding = tokenizer.encode(prompt)?;
        let tokens = encoding.ids;
        let eos_token = tokenizer.eos_token_id().unwrap_or(2);

        let mut stats = GenerationStats::new(tokens.len());
        stats.start();
        let target_logits = target.prefill(&tokens)?;
        let _ = draft.prefill(&tokens)?;
        stats.end_prefill();

        let mut sampler = crate::Sampler::new(crate::SamplerConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        });
        let next_token = sampler.sample(&target_logits, &tokens)?;
        let decoder = tokenizer.decode_stream(&tokens, true);
        let pos = tokens.len();

        Ok(Self {
            target,
            draft,
            sampler,
            config,
            stats,
            decoder,
            pos,
            next_token: Some(next_token),
            eos_token,
            all_tokens: tokens,
            finished: false,
            draft_k,
            context_length: None,
        })
    }

    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    pub fn next(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.finished {
            return Ok(None);
        }

        let current_token = match self.next_token {
            Some(t) => t,
            None => return Ok(None),
        };

        if current_token == self.eos_token || self.stats.generated_tokens >= self.config.max_tokens {
            self.finished = true;
            return Ok(self.decoder.flush()?);
        }

        self.all_tokens.push(current_token);
        self.stats.record_token();
        let text = self.decoder.step(current_token)?;

        let _ = self.draft_k;
        let _ = self.draft.forward(&[current_token], self.pos)?;
        let logits = self.target.forward(&[current_token], self.pos)?;
        self.pos += 1;
        self.next_token = Some(self.sampler.sample(&logits, &self.all_tokens)?);
        Ok(text.or(Some(String::new())))
    }

    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }
}
