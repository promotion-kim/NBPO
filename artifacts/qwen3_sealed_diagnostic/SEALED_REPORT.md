# P1 sealed reward report

Selected RONPO variant (locked before sealed access): `ronpo_k_only` (`top-mass`).

The corrected stability rule was finalized before reward scoring. DPO was excluded because record 252 has a genuine 1,163-token repeat run.

| Rank | Model | Worst (95% CI) | Avg | Win vs base | Helpfulness | Safety | Conciseness | Stability |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | inpo_avg | 0.2351 [0.2158, 0.2550] | 0.5085 | 51.57% | 0.5143 | 0.5079 | 0.5034 | passed |
| 2 | ronpo_full_expect | 0.2350 [0.2152, 0.2550] | 0.5042 | 50.00% | 0.5110 | 0.4995 | 0.5022 | passed |
| 3 | ht_mnpo_safety | 0.2337 [0.2139, 0.2540] | 0.5191 | 51.71% | 0.5222 | 0.5264 | 0.5088 | passed |
| 4 | sppo_avg | 0.2263 [0.2067, 0.2473] | 0.5137 | 51.79% | 0.5096 | 0.5273 | 0.5042 | passed |
| 5 | ronpo_k_only | 0.2227 [0.2041, 0.2417] | 0.4992 | 49.78% | 0.4772 | 0.5185 | 0.5019 | passed |
| 6 | simpo | 0.2217 [0.2020, 0.2420] | 0.5071 | 51.55% | 0.5056 | 0.5173 | 0.4984 | passed |
| 7 | ipo | 0.2205 [0.2014, 0.2400] | 0.5037 | 50.97% | 0.5050 | 0.4976 | 0.5085 | passed |
| 8 | ht_mnpo_conciseness | 0.2192 [0.1998, 0.2389] | 0.4987 | 49.86% | 0.5350 | 0.4832 | 0.4779 | passed |
| 9 | ht_mnpo_helpfulness | 0.2184 [0.1993, 0.2377] | 0.4979 | 50.41% | 0.5057 | 0.4908 | 0.4971 | passed |
| 10 | base | 0.2164 [0.1974, 0.2351] | 0.4939 | -- | 0.4777 | 0.4959 | 0.5081 | passed |
| -- | dpo | -- | -- | -- | -- | -- | -- | failed: repeat run 1,163 at index 252 |

## Provenance

- Prompt count: 604
- Sealed prompt SHA-256: `52b4028bd3ce095524e3ae66f49bf495d1236fea4635248b4263f9db1920df69`
- No sealed decoding was run during this resume. The preserved generations listed below were reused.
- Decode: vLLM; seed 42; temperature 0.7; top-p 0.9; max_new_tokens 2048; chat template; enable_thinking=false; bfloat16.
- Reward model: `RLHFlow/ArmoRM-Llama3-8B-v0.1@eb2676d20da2f2d41082289d23c59b9f7427f955`.
- Heads: `ultrafeedback-helpfulness`, `beavertails-is_safe`, and negated `helpsteer-verbosity`.
- Normalization: per-prompt min-max over the ten corrected-gate-eligible sealed models.
- Intervals: 2,000-resample paired prompt bootstrap, seed 42.
- Gate audit: `../gate_correction.json`; original failed gate JSON files are preserved.
- W&B run ID: `06d4f9eb494d` (https://wandb.ai/promotion-kim/mnpo/runs/06d4f9eb494d)
- Exact model revisions and generation SHA-256 values:
  - `base`: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`; generation `bdaaa84529d99321ef9ab1e1c86b7815514725b3ac034fd70484be5b0b8cbd72`
  - `ronpo_full_expect`: `promotion/qwen3-8b-aaai27-flagship-ronpo-full-expect-s42@17a1f1171627d43257182277e011e9b7b602ea53`; generation `fda9bb23d3483b94068a2a00ada6b655df1db9135852d8be9e1e67a73269b0f8`
  - `ronpo_k_only`: `promotion/qwen3-8b-aaai27-flagship-ronpo-k-only-s42@b8fdba53b2310b8b1f40079340138c3a5622df9f`; generation `33580848fc4eeb79829c1cbdeabb76ae79fad81b5cae9051c2f551b766292e40`
  - `dpo`: `promotion/qwen3-8b-aaai27-flagship-dpo-s42@edff1b2136635cb6ad639a4a5d00f074fdf6d946`; generation `ff570f00cb45b9ebf5edbc08620db82716358642618c6dabc3796e92740b6a27`
  - `ipo`: `promotion/qwen3-8b-aaai27-flagship-ipo-s42@cdb29724dbf62b13828e53463846d7449a25bc10`; generation `e9bd87be22b3b53a72f00205fe9a554720231e8458ebb07dd830e0287a5d9df7`
  - `simpo`: `promotion/qwen3-8b-aaai27-flagship-simpo-s42@7d9fa7b7c38737775af5e705c2da89000fb3b85f`; generation `c092a38f7ce62cc3c1ff2710a449e5f27f3758496699a0d25c04adcf89d11d9f`
  - `sppo_avg`: `promotion/qwen3-8b-aaai27-flagship-sppo-avg-s42@3e0da986b4fffc05d12ac7a069ad2b875cbc1dd7`; generation `44424c4fcbbe8adc3c5e2cdb9dc63bbc80c4d09cf3fbde4313d419df08224759`
  - `inpo_avg`: `promotion/qwen3-8b-aaai27-flagship-inpo-avg-s42@2ff371a07608f546211176b2ec6460126f2c41ea`; generation `1e89a17d1566a5d5eab21633d5fff2822840f06fdb6ad3ec203f883fe9dd0502`
  - `ht_mnpo_helpfulness`: `promotion/qwen3-8b-aaai27-flagship-ht-mnpo-helpfulness-s42@057b3dce0d42d9d97d3ebf4206dbed67c672d089`; generation `25051a23ac6449484fd6d78770d546fe8a0c9772ec8fb4419b77d00c408cb881`
  - `ht_mnpo_safety`: `promotion/qwen3-8b-aaai27-flagship-ht-mnpo-safety-s42@97e3d067e73e2bb3da8eefe243ef4a7052318385`; generation `90b4f4e8cba980b476d89f1ffe44a9bf2eaaf2e4d0774ef45e2782a95c96d61e`
  - `ht_mnpo_conciseness`: `promotion/qwen3-8b-aaai27-flagship-ht-mnpo-conciseness-s42@e26d1c0940505c3869f21a168911ce49669a973a`; generation `ab2bd242326f88544003c1ec4c60c3f6b10e5dc2ce5bc87049a0312dca90ffc5`
