#!/bin/sh
echo "Downloading example datasets from zenodo ( https://doi.org/10.5281/zenodo.53764 ) ..."
wget https://zenodo.org/record/53764/files/MM_Cglutamicum_D_aceE.ome.tiff
wget https://zenodo.org/record/53764/files/MM_Cglutamicum_SOS_HF.tif
echo "Verifying file integrity ..."
echo "6d4dc99ed4398f8e84344a5c2813d624 MM_Cglutamicum_D_aceE.ome.tiff" | md5sum -c -
echo "7ce3aa9895b76470ced975856df9d238 MM_Cglutamicum_SOS_HF.tif" | md5sum -c -

