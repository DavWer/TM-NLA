# Technical Writeup

## Motivation

Natural Language Autoencoders suggest a useful interpretability pattern: map internal activations into a language-facing space, then reconstruct the original activation from text to test whether the text carried the relevant information.

TM-NLA, short for Temporal Multimodal Natural Language Autoencoder, adapts that idea to video. Instead of probing only text-model residual states, it probes frozen visual-temporal hidden states from a vision-language model. It is an NLA-inspired temporal probe, not a full replication of Anthropic NLA.

## Method

For each video, TM-NLA samples a small sequence of temporal windows. Each window is passed through a frozen `Qwen/Qwen3.5-0.8B` vision-language model, and visual-token hidden states are pooled from a target layer.

The learned system has three parts around the frozen base model:

- a temporal probe / contextualizer,
- an activation verbalizer, AV,
- a text reconstructor, AR.

The temporal contextualizer consumes neighboring hidden states and produces a contextual state for the center window. A learned adapter projects that contextual state into the language model's embedding pathway, producing a continuous language-manifold activation.

Textual externalization is then split into two learned components:

- **Activation Verbalizer, AV:** receives a probed activation vector and conditions the frozen language model to generate a small set of short candidate phrases. These phrases are not treated as ground truth captions; they are readout hypotheses.
- **Text Reconstructor, AR:** receives a candidate phrase and maps it back into the activation space. This reconstructed activation is compared against the original target activation and against a control activation from another window.

The text reconstructor is used as a compatibility verifier/reranker. The selected readout is the candidate with the strongest specificity margin against the control activation. The public `specificity_status` field is derived only from activation metrics, not phrase vocabulary.

## Temporal Addition

The main extension beyond the original NLA framing is temporal state. The probe operates over video windows and preserves per-window records:

- video id,
- timestamp,
- window index,
- center hidden state,
- contextual hidden state,
- projected language-manifold embedding,
- optional thumbnail.

This allows the readout to be organized as a semantic trajectory rather than a single static phrase.

## Results

The geometry result is the strongest part of the project. Temporal contextualization showed improvements in locality and transition-like structure, and the selected full-precision probe preserved useful activation geometry across several small video domains.

The text result is more limited. Soft compatibility training improved the reconstructor's ability to distinguish compatible captions from incompatible random captions, but exact caption ranking remained ambiguous because many captions are valid paraphrases or nearby descriptions.

Single no-sampling AV decoding was tested and often failed on reasonable clips, so the public readout uses activation-conditioned candidate generation plus AR reranking rather than pretending the AV is a solved direct decoder.

The final public demos show this as weak temporal semantic traces rather than robust captions: repeated ball/player/court fragments in a basketball clip, and horse/field/running fragments in a horseback-riding clip.

## Final Conclusion

TM-NLA is a completed proof of concept for visual-temporal language-manifold probing. It shows weak recoverable temporal semantic traces from frozen visual-temporal activations. It is not a robust video captioning system, classifier, production video-understanding tool, full Anthropic NLA replication, or solved activation-to-language decoder.
