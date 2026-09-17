# Third-party software

The SCALE model and dataset compatibility code are adapted from the upstream SongFormer repository and retain its Creative Commons Attribution 4.0 license in `LICENSE`.

`third_party/soulx_singer` contains the SoulX-Singer preprocessing components used for lyric extraction, distributed under Apache License 2.0. Its original license is preserved at `third_party/soulx_singer/LICENSE`.

Pretrained MuQ, Longformer, XLM-R, SoulX preprocessing, SaT, and SCALE checkpoint files are not distributed in this repository. Their upstream terms apply separately.

The vendored subset excludes the MIDI editor, note transcription, and singing synthesis components. Its package exports are limited to the preprocessing components used here.
