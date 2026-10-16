# VLA inference source

Extracted from Hippocampus revision `e998ec8a1584bdc8de6d2be87fe4574a823233ab` (master). Original source locations:
`LlamaFactory/src/ctx_to_lora`, `LlamaFactory/src/llamafactory/train/vla`,
`LlamaFactory/src/llamafactory/train/c2l_vla`, and `evaluation/simpler`.
The release separates module namespaces, replaces pickle metadata loading with typed
JSON reconstruction, removes training entrypoints and unused action decoders,
and adds portable evaluation orchestration. The numerical inference implementation
is retained. See the root `NOTICE` and `LICENSES/` for upstream license terms.
The `ctx_to_lora/modeling/idefics2.py` header retains its original Apache notice.
