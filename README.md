# Spatial analysis of human and nonhuman primate basal ganglia
This repository contains analysis and figure generation code supporting the publication: 

> Hewitt, Turner, ... Close, Long, Seeman, et al., Spatial transcriptomics reveals molecular organization of human and nonhuman primate basal ganglia, Cell (2026), https://doi.org/10.1016/j.cell.2026.10.023.

This publication is part of a wider [Human and Mammalian Brain Atlas (HMBA)](https://brain-map.org/consortia/hmba) effort within the NIH [BRAIN Initiative Cell Atlas Network (BICAN)](https://brain-bican.org/) consortium. It is part of a Cell Press publication package describing the scientific findings associated with the [inaugural HMBA data release](https://brain-map.org/consortia/hmba/hmba-release-basal-ganglia) covering the Basal Ganglia.

## Summary
The basal ganglia (BG) are interconnected subcortical nuclei with a complex topographical organization that orchestrate goal-directed behaviors and are implicated in neurodegenerative movement disorders. We generated a cellular-resolution spatial transcriptomics atlas of the BG in human, rhesus macaque, and common marmoset, sampling over one million cells in each species. By integrating spatial data with a cross-species consensus single-nucleus RNA sequencing (snRNA-seq) cell-type taxonomy, this atlas reveals principles of molecular organization within and across structures. Extensive spatial sampling across the rostral-caudal axis illuminates complex, yet highly stereotyped, multi-axis gene expression gradients superimposed onto discrete compartments of the striatum. Additionally, we identify core-shell organization of the internal globus pallidus. This unified, cross-species spatial transcriptomics atlas will be a foundational resource for characterizing the molecular and functional organization of the BG and their roles in health and disease.


<img width="600" height="600" alt="HewittTurner_Graphical_Abstract_300DPI" src="https://github.com/user-attachments/assets/905dd274-598f-49a6-8fc9-1bf5fb7153b7" />

## Repository organization
Under construction

## Raw Data Sources
Raw transcript image data for the two MERSCOPE (human, macaque) and one Xenium (marmoset) spatial transcriptomics datasets from this publication can be found at the Brain Image Library (BIL): [https://doi.org/10.35077/g.1211](https://doi.org/10.35077/g.1211).

## Processed Data Sources
Pre-processed data artifacts helpful for running the analyses in this repository are available at the public AWS S3 bucket: [s3://allen-hmba-releases/](https://allen-hmba-releases.s3.us-west-2.amazonaws.com/index.html):
- Data 1
- Data 2
- etc.

Analyses in this repository draw on spatial transcriptomics and single-cell transcriptomics that can also be accessed programmatically via AWS S3, which is detailed in these [Jupyter notebook tutorials](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-BG_dataset.html).

## Interactive Data Visualization
You can interactively explore [all three species' spatial transcriptomics datasets side-by-side](https://knowledge.brain-map.org/abcatlas?_gl=1*16da9he*_gcl_au*MTYyMjE0ODkwNS4xNzg3NjAwNjA2Li0uLS4xNzg3NjAwNjA2LjE3NDU2NTE2OTQuMTc5MzM4MTk5Mi4xNzkzMzg0MDkx*_ga*MTAwNDIxNjAxLjE3NjkxMDY5OTY.*_ga_MFZ8RZNMY4*czE3OTMzODE5OTIkbzUwJGcxJHQxNzkzMzg0Mjc0JGo1OSRsMCRoMjYzNDU3ODk2#AQMBM0NQSVAyODA0U0JEWDFEWDFSVwACUzVLUVRGSENSS0hWWlZSUjVIUgADBQFFM0hHQllIUkM5OEdBNkZXOUY2AAIETm9ubmV1cm9uAEdsdXQgU2VybyBEb3BhAFN1YnBhbGxpdW0gR0FCQQBTdWJwYWxsaXVtIEdBQkEtR2x1dAAAATNYQk9GQlIzMDJFWUMwVVFENTEAAgAAAVNRNldKTzBHTlVKR0ZVOExIVlYAAgAAAUhLMTRPMUFTQU5QWUhPTkZMTFEAAgAAATZFV0JTRUo0WUZWR0tJMkFQQkwAAgAABAEAAoVsLWWGxlUGA4fMF4eGLt%2FTBFdNUDhOTUo1RDhYTzFBT0kxTDkABYHYoeuEthGZhNqUXYUChoYGAAcAAAUABgEBAkhLMTRPMUFTQU5QWUhPTkZMTFEAA34AAAAEAAAISFpFWVhTUU9FRE5EMlE2TTk3TQAJUE9aMkhDUEJUNjBEU0RKOFVBNwAKAAsBNlQ5NjI2MDEyNlRWNEtMR0RYRwACQ1JFSFJDSzJZWFJXSksySjBBTAADAQQBAAIjMDAwMDAwAAPIAQAFAQECIzAwMDAwMAADyAEAAAABVk00RTZKOURQWVlEN0UwQkhOQgACS0daTEFXSVBDUjRXOThGNUxENgADBQFFM0hHQllIUkM5OEdBNkZXOUY2AAIETm9ubmV1cm9uAEdsdXQgU2VybyBEb3BhAFN1YnBhbGxpdW0gR0FCQQBTdWJwYWxsaXVtIEdBQkEtR2x1dAAAATNYQk9GQlIzMDJFWUMwVVFENTEAAgAAAVNRNldKTzBHTlVKR0ZVOExIVlYAAgAAAUhLMTRPMUFTQU5QWUhPTkZMTFEAAgAAATZFV0JTRUo0WUZWR0tJMkFQQkwAAgAABAEAAoU4tniFxGEGA4XCiuuGuAYmBFdNUDhOTUo1RDhYTzFBT0kxTDkABYQUxBaDjMVShGJpv4S050sGAAcAAAUBAVRNRU0xMzJDAAAGAQECSEsxNE8xQVNBTlBZSE9ORkxMUQADfgAAAAQAAAg5QVVURTlTMTBLQUVDMlVaTzMxAAlQT1oySENQQlQ2MERTREo4VUE3AAoACwFZM1AxSk0xRU5DR1dFVzhVU1RWAAJDUkVIUkNLMllYUldKSzJKMEFMAAMBBAEAAiMwMDAwMDAAA8gBAAUBAQIjMDAwMDAwAAPIAQAAAAFOWEo5TlpBUUE1TUVIUU5PV0pRAAJUR0JRMkZFVjI3TjBDUExLVkFaAAMBAUUzSEdCWUhSQzk4R0E2Rlc5RjYAAgROb25uZXVyb24AU3VicGFsbGl1bSBHQUJBAEdsdXQgU2VybyBEb3BhAFN1YnBhbGxpdW0gR0FCQS1HbHV0AAAEAQAChCD6lYS6JVADhagoK4V25WUEV01QOE5NSjVEOFhPMUFPSTFMOQAFg2xvMYI24kGDwAnEgzgCUgYABwAABQAGAQECSEsxNE8xQVNBTlBZSE9ORkxMUQADfgAAAAQAAAhJQTZZNlNON1FaU0lFSjBGWjBVAAlQT1oySENQQlQ2MERTREo4VUE3AAoACwE3NDE2QjVYSkRTWTBIREY5M1JPAAJDUkVIUkNLMllYUldKSzJKMEFMAAMBBAEAAiMwMDAwMDAAA8gBAAUBAQIjMDAwMDAwAAPIAQAAAAIHAA%3D%3D) 
(including BG Taxonomy cell type localization, spatial gene expression, and manual anatomical annotations) at the Allen Brain Cell (ABC) Atlas, a web browser-based platform that allows users to visualize multiple multimodal single cell data datasets simultaneously.

To can also use the ABC Atlas platform to view the companion, cross-species [BG single-cell transcriptomics dataset](https://knowledge.brain-map.org/abcatlas?_gl=1*nbbtsk*_gcl_au*MTYyMjE0ODkwNS4xNzg3NjAwNjA2Li0uLS4xNzg3NjAwNjA2LjE3NDU2NTE2OTQuMTc5MzM4MTk5Mi4xNzkzMzg0MDkx*_ga*MTAwNDIxNjAxLjE3NjkxMDY5OTY.*_ga_MFZ8RZNMY4*czE3OTMzODE5OTIkbzUwJGcxJHQxNzkzMzg0Mjc0JGo1OSRsMCRoMjYzNDU3ODk2#AQEBRUlVQUxQWU5HWjVWVTdZMlkwMgACUVpRMlFCTVU0VDVJNDdEMUs0RgADAAQBAQKBItJbgQi1PgODUGd%2Bg%2Bp7yQAFAAYBAQJISzE0TzFBU0FOUFlIT05GTExRAAN%2BAAAABAAACEZUTEFVT1Y2U0xCMDcxU1ZOVTkACVBPWjJIQ1BCVDYwRFNESjhVQTcACgALAW5vbmUAAm5vbmUAAwEEAQACIzAwMDAwMAADyAEABQEBAiMwMDAwMDAAA8gBAAAAAgEA).

Both modalities are annotated with the Mammalian Basal Ganglia Consensus Cell Type Taxonomy.

## Companion Publications
- Fu, Y., Johansen, N.J., Kempynck, N., Ding, W., Turner, M.A., Garcia, A.D., Schmitz, M.T., Close, J., Kapen, I., Hewitt, M., et al. (2025). Circuit specific specialization of human basal ganglia astrocytes.
- 

## Code availability and reproducibility
Code is provided for transparency and reproducibility of the analyses described in the associated manuscript. Execution of some workflows may require access to large datasets and high-performance computing resources.

## Level of support
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response. We are not expecting to accept external contributions to this code.

## Contact
For questions or issues related to this repository, please contact the corresponding authors listed in the publication.

## Citation
If you use this code, or components or modifications thereof, in your work, please cite the publication:

> M.N. Hewitt, M.A. Turner, et al., Spatial transcriptomics reveals molecular organization of human and nonhuman primate basal ganglia, Cell (2026), https://doi.org/10.1016/j.cell.2026.10.023.






