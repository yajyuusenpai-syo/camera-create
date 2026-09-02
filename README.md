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
`docs/THREE_ENV_SETUP.md`.

Checkpoint layout:

```text
ckpt/
  pi3x/
  moge3/
  vipe/
```

VIPE is source code plus its own downloaded model cache rather than one single
checkpoint. `ckpt/vipe/` is reserved for an optional explicit VIPE cache.

The target deployment uses three isolated Python 3.10 environments (Pi3X,
MoGe-3, and VIPE). The current implementation still contains the original
in-process MoGe-2 backend; see `docs/DEPLOYMENT.md` for the migration status and
do not treat the three-environment path as GPU-validated yet.
