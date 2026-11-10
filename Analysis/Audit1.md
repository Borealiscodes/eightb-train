This is not authoritative it looks at potential issues as I attempt research consolidation and tooling to build out of band geometric adapters. It is a blind audit
---

eightb-train is a significant architectural pivot away from custom runtime math engines toward a standard QLoRA fine-tuning pipeline wrapped in a post-hoc ternary weight sealer.
Rather than fixing the out-of-order operator updates or runtime manifold collapse we diagnosed in Vectorium and smart-tree, the 8b-is team has shifted to training a dense 27B model (Qwen/Qwen3.8-27B) using standard Hugging Face tools (transformers + peft) and compressing the merged weights afterward (pqseal.py). [1] (/goto?url=CAESxQEBxCJF1yLdggEdHnnMG7bTab-0yk4gkR1tZ7-FxzixfIRIZ1OcKo9LU7Fro_I5E-TqT7cHm3bfksKnvckT6LoZMf9SvQN3C-9yFiIS0VBAC9JhNfGTQ-bO5JhfOBbbgefYQBrEYF4-Gqyq2kaMptSfIee_TPS6_iOqdNI7oleKnASEyawmGvAwq7BMtFqY2xlWbP4LX085ZtUfvDOTZnCxHr3Nf-oa8z_8Lm1mg_z6QlKruLx7rPPeydkWcSsUI-BUMuk2_g)
Here is the technical audit of what changed, what improved, and where the exact same mathematical vulnerabilities remain hidden.

1. What Changed (And What Actually Improved)

* Standardized SFT Core (train.py & merge_adapter.py): They abandoned custom runtime differential operators for training and moved to standard 4-bit NF4 QLoRA Supervised Fine-Tuning on an A6000 (48 GiB), followed by a bf16 adapter merge. This eliminates custom gradient-ordering bugs during the training phase. [1] (/goto?url=CAESsgEBxCJF1xcCRUqQpzMKaSGbBEW3blJMd78wh_ib6GRNbcBxCzShsICZweHSp0vKBROYl3XVyRxhdWAe_KdJJJwdulmnj-wXDxklr2tu9uXZy52RH5ejMB7hgTqWRytosHA8pXibtC4PtzkP_jJ9TVkesbVY0NojyvxRXUnvAxgSP2rNCdBN567Po23ixjXx6pWCxWD6uw-PAkDSNkBmp3QEGLFUVz2f8v2VQdlAd25nmogo)
* Honest Status Disclosures: Unlike earlier repos that claimed full verification over placeholder stubs, the README.md explicitly admits what is stubbed (vision / voice lanes), what ran on CPU vs. GPU, and why GGUF weights cannot be QLoRA'd directly. [1] (/goto?url=CAESxgEBxCJF1_6u9xOXFmPTI5sTQaxUbBWeQsjZPJ1FjzAowyXz_yOzez7nEtzPyeQHp6kqoOipzCkgUpF2Wk1Tm0oX7Fb6vgNnXkaNEMDhVc7ui4xKpfdShNk0xm4YmHUxudxIhBvQxaRxZdK06YKeOVg2GHH45xGYtdt9K8Jmc2J4DlCX7PXGO2JzB31DEJX0x5UzCk3QY8WJR4rZr8IHfcrCPzraua2KtOKKjDfVujEvHcaPW7crQGpOuQ3IC-Wn7m4lRRz2gZw)
* Clean Codec Separation (pqseal.py): They isolated their exact rational p/q quantizer, SPQT Base-243 packer (3⁵ = 243 ≤ 256), and 64-byte wire seal (64B@0x62) into a standalone post-merge step. [1] (/goto?url=CAES3gEBxCJF15yoVERY2mLbUkt65UxXmlGBD5vBX23gC1zHxTgbRWCf3WOkQwGWOm8ip__-GUdDDJ10pXWrm76IMfqnJCZPg2bkDeihRH2wHimt7gZUpXyLsbYhV6Pk8FOAjWNNWy2OG_XOgW_WBFDN0NkuEmnQxSajw-5_yTvdgC1yngsD_0P7tZBc43AirsGCKARJ6PG-LOZTYdquHfzvjWXF8ZP4KOsTFmoOTIoA_6aGsVN6iQcSyy8rgWJc264_EBPoumiAaYCiTTGedbUqzBJi-_GqOoOiEIC3BFNGPEU)

2. Where the Core Mathematical Vulnerabilities Persist


Component / Claim	What eightb-train Does	Underlying Mathematical Reality	Audit Verdict
Group-Wise Ternary Quantizer (pqseal.py)	Applies 2 Lloyd-Max steps (τ₀ = 68μ/100, τ = α/2) in integer Q16 to snap merged bf16 weights to \(\{-1, 0, +1\}\).	Snapping a dense 27B model to ternary weights after bf16 SFT injects a massive quantization error matrix E without quantization-aware training (QAT).	⚠️ Post-Hoc Rank Shock
"No Float Drift" Claim	Computes threshold scalars (τ, α) using Q16 fixed-point integers during weight packing.	Q16 prevents rounding drift inside the packer, but the downstream inference engine still runs floating-point activations over rank-compressed ternary weights (\(\sigma_{\min}(W + E) \to 0^+\)).	❌ Category Error
64B@0x62 Wire Seal & Disjoint Bitplanes	Splits weights into disjoint BFP-64 positive/negative planes (pos ∧ neg = 0) with a 16-bit checksum over the first 62 bytes.	Verifies bit-level storage integrity (identical to your Base243_Bijection.v and RuntimeFusion.v), but offers zero protection against semantic drift or attention collapse.	⚠️ Plumbing-Only Check
Stubbed Vision / Voice Lanes	Lists vision-ocr-pairs and dsp-audio-captions in config/base-27b.yaml while train.py executes text-only SFT.	Multi-modal alignment is unverified; bolting audio/vision collators onto a post-hoc ternary-sealed text checkpoint risks severe cross-modal manifold shear.	⚠️ Incomplete Contract


3. How This Connects to Your Verified Stack

* Exact Codec Alignment: Their pqseal.py pipeline (quantize_groupwise_l2_ternary_pq, pack_trit_chunk, and 64B@0x62 bitplane splitting) is the exact Python counterpart to the Coq proofs you already locked down in Base243_Bijection.v and RuntimeFusion.v. [1] (/goto?url=CAESxgEBxCJF12LNZ2VR3FacQGFY1Liai6Jtip1Q2KnCciieb61g_wl2w2OQGfN6RE45rTPRweD1KLEl0l971fs6kjFmEHgFRyVjqIQRqcYmb341eTQP4v5KlkfLnoq1Q8OtqYyNZR8AE0SS71D75Na1-8ruKDWmOJqrV0SJGCpaz6bjNPRXdR0VuIcP8UVyGB5SI88R2EpBq5YN_L00QTpp8YkZoQixYBJcAXjZbp7Goa7_Nr5DPGbI4YlQrZi0H3batWz_Nfc43Zk)
* The Missing Triage Gate: Because eightb-train applies aggressive post-training ternary quantization (Q16 Lloyd-Max) after merging bf16 weights—without measuring singular value collapse—the resulting weight decks are prime candidates for Altitude A7 metric degeneracy (\(\sigma_{\min} \to 0^+\)). Running your A7.2-CollapseSurfaceRepairPipeline (collapse_pipeline.py + NonDualProjector.jl) on the sealed weight blocks would immediately expose how many dimensions were sheared during the bf16 → ternary snap.
If you want to inspect the actual implementation details inside Borealiscodes/eightb-train, we can:
* Audit pqseal.py and mm.py directly to check their Q16 Lloyd-Max math and 64B@0x62 bitplane logic against your Coq Base243 and RuntimeFusion proofs
* Inspect train.py and config/base-27b.yaml to check the dataset weighting mix and QLoRA hyperparameters for stability risks
