# hash:sha256:faac9f21a67de35fc18573ad427cfda3862cfcba53271e19b052644c4b32114c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ttf-mscorefonts-installer=3.8ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        mscorefonts==0.0.1 \
    && conda clean -ya

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    anndata==0.12.19 \
    cmap==0.8.0 \
    colorcet==3.2.1 \
    nibabel==5.4.2 \
    opencv-python-headless==5.0.0.93 \
    pandas==2.3.3 \
    plotly==7.1.0 \
    scikit-misc==0.5.3 \
    spatialdata[extra]==0.8.0 \
    xarray==2026.9.0 \
    -e git+https://github.com/AllenInstitute/code-ocean-utils-mnh.git@88043fd69776c1da858c5ef2e85875283ebf82de#egg=code-ocean-utils-mnh \
    -e git+https://github.com/AllenInstitute/spatial_compare.git@b638b2e5835c35dc1aa3a4653f6a428d270d97df#egg=spatial-compare \
    -e git+https://github.com/AllenInstitute/spots-in-space.git@485fd05a6f58d32f8c881ee99d0460ddb1ecccc1#egg=spots-in-space
