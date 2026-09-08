# test32-CGMD

This repository runs the generation/CG export stage from an already trained
`test32.py` checkpoint. It does not train the model.

## Supercomputer setup

Keep this repository, DM2, and the checkpoint directory under one parent:

```text
work/
├── test32-CGMD/
├── DM2/
└── checkpoints/
    └── test32_sio2_glass_nequip.pt
```

Build the container from the `test32-CGMD` directory:

```bash
singularity build --force --fakeroot \
  test32-cgmd-pytorch-2.5.0-cu124.sif \
  Singularity.test32-CGMD.def
```

Rebuild the image with this command after pulling dependency changes. An
existing image does not change when the definition file is updated.

Replace `<ProjectGroup_ID>` with the ID reported by the site's `listu`
command and submit from the repository directory:

```bash
qsub -P <ProjectGroup_ID> run_test32-CGMD.pbs
```

The PBS script automatically finds the sibling `DM2` and `checkpoints`
directories. For another layout, provide absolute paths:

```bash
qsub -P <ProjectGroup_ID> \
  -v DM2_ROOT=/path/to/DM2,CHECKPOINT_PATH=/path/to/test32_sio2_glass_nequip.pt \
  run_test32-CGMD.pbs
```

`INPUT_PATH`, `SIF_IMAGE`, and `OUTPUT_DIR` can be overridden in the same
way. External input/output directories are added to the container bind mounts
automatically. Both SingularityCE and Apptainer are supported.

Generation restart data is written periodically. If a job reaches its time
budget, submit the same command again to resume.
