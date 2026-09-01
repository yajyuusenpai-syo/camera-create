# camera-create

Camera annotation pipeline using the SANA-WM Stage 2 method for end-to-end
metric camera estimation from a monocular video:

```text
video -> Pi3X temporal depth -> MoGe-2 metric anchor -> EMA fusion
      -> VIPE bundle adjustment -> metric intrinsics/extrinsics + validation
```

The primary entry point is `cli.py`; the package also installs the
`camera-create` command. See `docs/DEPLOYMENT.md` for installation and usage.

Checkpoint layout:

```text
ckpt/
  pi3x/
  moge2/
  vipe/
```

VIPE is source code plus its own downloaded model cache rather than one single
checkpoint. `ckpt/vipe/` is reserved for an optional explicit VIPE cache.
