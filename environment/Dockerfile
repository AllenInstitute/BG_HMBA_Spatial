# hash:sha256:6a79ac641ee204a56b0a8d89af5385b724f873a11587e5feb53f6cf1734abfd4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        ipykernel \
        ipywidgets \
        jupyterlab \
        notebook \
        pyyaml \
    && conda clean -ya

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    cmap \
    colorcet \
    nibabel \
    opencv-python-headless \
    plotly \
    scikit-misc \
    xarray \
    -e git+https://github.com/AllenInstitute/code-ocean-utils-mnh.git@88043fd69776c1da858c5ef2e85875283ebf82de#egg=code-ocean-utils-mnh \
    -e git+https://github.com/AllenInstitute/spatial_compare.git@b638b2e5835c35dc1aa3a4653f6a428d270d97df#egg=spatial-compare \
    -e git+https://github.com/AllenInstitute/spots-in-space.git@485fd05a6f58d32f8c881ee99d0460ddb1ecccc1#egg=spots-in-space
