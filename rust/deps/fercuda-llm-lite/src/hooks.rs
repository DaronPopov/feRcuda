use candle_core::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct LogitsCandidate {
    pub token_id: u32,
    pub logit: f32,
}

pub trait LogitsHook: Send {
    fn rewrite_top_logits(
        &mut self,
        candidates: &[LogitsCandidate],
        generated_tokens: &[u32],
    ) -> Result<Option<Vec<LogitsCandidate>>, String>;
}

pub(crate) fn top_logits_candidates(
    logits: &Tensor,
    limit: usize,
) -> Result<Vec<LogitsCandidate>, String> {
    let squeezed = squeeze_logits(logits)?;
    let values: Vec<f32> = squeezed
        .to_vec1()
        .map_err(|e| format!("failed to read logits tensor: {e}"))?;

    let mut indexed: Vec<(usize, f32)> = values.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(indexed
        .into_iter()
        .take(limit.max(1))
        .map(|(token_id, logit)| LogitsCandidate {
            token_id: token_id as u32,
            logit,
        })
        .collect())
}

pub(crate) fn apply_candidate_rewrite(
    logits: &Tensor,
    rewritten: &[LogitsCandidate],
) -> Result<Tensor, String> {
    let dims = logits.dims().to_vec();
    let squeezed = squeeze_logits(logits)?;
    let mut values: Vec<f32> = squeezed
        .to_vec1()
        .map_err(|e| format!("failed to materialize logits tensor: {e}"))?;

    for candidate in rewritten {
        let index = candidate.token_id as usize;
        if index < values.len() {
            values[index] = candidate.logit;
        }
    }

    match dims.as_slice() {
        [n] => Tensor::from_vec(values, *n, logits.device())
            .map_err(|e| format!("failed to rebuild logits tensor: {e}")),
        [m, n] => Tensor::from_vec(values, (*m, *n), logits.device())
            .map_err(|e| format!("failed to rebuild logits tensor: {e}")),
        _ => Err(format!("unsupported logits rank for hook rewrite: {:?}", dims)),
    }
}

fn squeeze_logits(logits: &Tensor) -> Result<Tensor, String> {
    match logits.dims() {
        [1, _] => logits
            .squeeze(0)
            .map_err(|e| format!("failed to squeeze logits tensor: {e}")),
        [_] => Ok(logits.clone()),
        dims => Err(format!("unsupported logits tensor dims: {:?}", dims)),
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_candidate_rewrite, top_logits_candidates, LogitsCandidate};

    #[test]
    fn top_logits_candidates_orders_descending() {
        let logits =
            candle_core::Tensor::from_vec(vec![0.1f32, 2.5, 1.0, 3.0], 4, &candle_core::Device::Cpu)
                .unwrap();
        let top = top_logits_candidates(&logits, 2).unwrap();
        assert_eq!(
            top,
            vec![
                LogitsCandidate {
                    token_id: 3,
                    logit: 3.0
                },
                LogitsCandidate {
                    token_id: 1,
                    logit: 2.5
                },
            ]
        );
    }

    #[test]
    fn apply_candidate_rewrite_updates_selected_logits() {
        let logits =
            candle_core::Tensor::from_vec(vec![0.1f32, 2.5, 1.0, 3.0], 4, &candle_core::Device::Cpu)
                .unwrap();
        let rewritten = vec![LogitsCandidate {
            token_id: 1,
            logit: 9.0,
        }];
        let updated = apply_candidate_rewrite(&logits, &rewritten).unwrap();
        let vec = updated.to_vec1::<f32>().unwrap();
        assert_eq!(vec, vec![0.1, 9.0, 1.0, 3.0]);
    }
}
