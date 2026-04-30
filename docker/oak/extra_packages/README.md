# OAK / OAK4 SDK drop-in directory

Place the licence-gated SDK archives here before running
`docker/oak/build.sh`. Both archives must be present (with the exact
filenames below) for the corresponding image to build. They are
**gitignored** — never commit them.

## RVC2 / OAK — OpenVINO 2022.3.0 dev archive

**File expected:** `openvino-2022.3.0.tar.gz`

Download from the OpenVINO archive:
<https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/>

Pick the Linux dev archive (e.g.
`l_openvino_toolkit_dev_ubuntu20_p_2022.3.0.<build>.tgz`) and rename
it to `openvino-2022.3.0.tar.gz`.

## RVC4 / OAK4 — Qualcomm Neural Processing SDK (SNPE)

**File expected:** `snpe-2.32.6.zip`

Download from Qualcomm's Software Center (free account required;
licence acceptance is mandatory):
<https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community>

Rename the archive to `snpe-2.32.6.zip`. To pin a different SNPE
version, override `SNPE_VERSION` when invoking `build.sh`:

```bash
SNPE_VERSION=2.34.0 docker/oak/build.sh rvc4
```

(the file you drop in must then be named `snpe-2.34.0.zip`).

## Build

Once both archives are in place:

```bash
docker/oak/build.sh         # builds both rvc2 and rvc4 from scratch
docker/oak/build.sh rvc2    # rvc2 only
docker/oak/build.sh rvc4    # rvc4 only
```

This produces:

- `luxonis/modelconverter-rvc2:local`
- `luxonis/modelconverter-rvc4:local`

which match the default tags in `train_and_export.py`. Override with
`--oak-rvc2-image` / `--oak-rvc4-image` to use different tags.
