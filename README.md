# camera-create

Camera annotation pipeline using the SANA-WM Stage 2 method for end-to-end
metric camera estimation from a monocular video:

```text
video -> Pi3X temporal depth -> MoGe-3 metric anchor -> EMA fusion
      -> VIPE bundle adjustment -> metric intrinsics/extrinsics + validation
```

The primary entry point is `cli.py`; the package also installs the
`camera-create` command. See `docs/DEPLOYMENT.md` for installation and usage.
For exact source cloning and isolated environment commands, see
`docs/THREE_ENV_SETUP.md`. Both standard `venv` and Conda-prefix installation
scripts are provided; they create the same `.envs/<model>` runtime layout.
Recursive multi-GPU directory scheduling, per-worker checkpoints, resume, tqdm,
and `cam_<video name>.json` output are documented in `docs/BATCH_PROCESSING.md`.

Checkpoint layout:

```text
ckpt/
  pi3x/
  moge3/
  vipe/
```

VIPE is source code plus its own downloaded model cache rather than one single
checkpoint. `ckpt/vipe/` is reserved for an optional explicit VIPE cache.

The pipeline invokes Pi3X, MoGe-3, and VIPE through three isolated Python 3.10
environments and exchanges validated NPZ caches between processes. See
`docs/DEPLOYMENT.md`; real-model GPU accuracy still requires server validation.
