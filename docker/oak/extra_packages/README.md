# OAK / OAK4 SDK drop-in directory

Place the licence-gated SDK archives here before running
`docker/oak/build.sh`. Both archives must be present (with the exact
filenames below) for the corresponding image to build. They are
**gitignored** — never commit them.

> **Architecture.** The upstream Luxonis Dockerfiles hard-code
> `/usr/lib/x86_64-linux-gnu/` library paths, and these SDKs only ship
> x86_64 Linux binaries. Both archives **must be the x86_64 Linux
> variants**, and the resulting image only runs as `linux/amd64`. On
> ARM hosts (Apple Silicon, ARM Linux) the build script forces
> `--platform linux/amd64` so Docker emulates via QEMU. See the host
> architecture note in the top-level README.

## RVC2 / OAK — OpenVINO 2022.3.0 dev archive

**File expected:** `openvino-2022.3.0.tar.gz`

Download the **Linux x86_64 dev archive** (NOT the runtime build, NOT
the Windows or macOS builds, NOT a non-Ubuntu Linux build):

<https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/>

Pick `l_openvino_toolkit_dev_ubuntu20_p_2022.3.0.<build>.tgz` (the
filename includes a build suffix like `.9052`). Then rename:

```bash
mv l_openvino_toolkit_dev_ubuntu20_p_2022.3.0.*.tgz openvino-2022.3.0.tar.gz
```

(`.tgz` and `.tar.gz` are the same gzip-compressed tar — only the
extension matters; the upstream Dockerfile expects `.tar.gz`.)

The dev archive ships `mo` (Model Optimizer) and `compile_tool`, both
of which the RVC2 converter invokes. The runtime archive lacks them
and will silently break the build.

## RVC4 / OAK4 — Qualcomm Neural Processing SDK (SNPE)

**File expected:** `snpe-2.32.6.zip`

Download the **Linux x86_64 SDK** (the catalog also lists Windows and
Android-only variants — those will not work):

<https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community>

A free Qualcomm developer account and licence acceptance are
required. The download is typically named something like
`v2.32.6.250402.zip` or contains an inner `qairt/` directory; rename
to `snpe-2.32.6.zip`:

```bash
mv qualcomm_neural_processing_sdk_v2.32.6.*.zip snpe-2.32.6.zip
```

The archive must contain `bin/x86_64-linux-clang/` and
`lib/x86_64-linux-clang/` directories — the upstream Dockerfile
prunes the Windows / Android / Hexagon / Ubuntu variants and keeps
the x86_64-linux-clang one.

To pin a different SNPE version, override `SNPE_VERSION` when
invoking `build.sh`:

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

- `luxonis/modelconverter-rvc2:local`  (linux/amd64)
- `luxonis/modelconverter-rvc4:local`  (linux/amd64)

which match the default tags in `train_and_export.py`. Override with
`--oak-rvc2-image` / `--oak-rvc4-image` to use different tags.
