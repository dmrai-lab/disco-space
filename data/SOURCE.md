# DiSCo reference files

From the Diffusion-Simulated Connectivity (DiSCo) dataset, Rafael-Patino J., Girard G., Truffet R., Pizzolato M.,
Caruyer E., Thiran J.-P., "The Diffusion-Simulated Connectivity (DiSCo) dataset", Data in Brief 38 (2021) 107429,
https://doi.org/10.1016/j.dib.2021.107429; data at https://doi.org/10.5281/zenodo.4634234. Licence CC BY 4.0.

| file | what |
|---|---|
| `DiSCo_mask.nii.gz` | the 40³ mask of the phantom's voxels (the tracking domain) |
| `DiSCo_ROIs.nii.gz` | the sixteen end regions, labels 1 to 16 (seeds and connectome endpoints) |
| `DiSCo_gradients.bvals`, `DiSCo_gradients_dipy.bvecs` | the dataset's 364-measurement gradient table (b in s/mm²), the "DiSCo 364" acquisition |
| `DiSCo_Connectivity_Matrix_Strands_Count.txt` | ground truth: strands per region pair |
| `DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt` | ground truth: total cross-sectional area per region pair |
| `DiSCo_Strands_Trajectories.tck`, `DiSCo_Strands_Diameters.txt` | the 12,196 strands' centerlines (in units of the 25 µm voxel, i.e. the image grid) and inner diameters (mm): the ground-truth view |

Unmodified copies of the dataset's DiSCo1 files, redistributed here under CC BY 4.0 so the Space is self-contained.
