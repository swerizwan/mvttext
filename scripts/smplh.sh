#!/bin/bash
echo "Extraction of the archives"
echo

cd reqs/smplh
mkdir tmp
cd tmp

tar xfv ../smplh.tar.xz
unzip ../mano_v1_2.zip

cd ../../../
echo
echo "Done!"
echo
echo "Clean and merge models"
echo

python scripts/merge_smplh_mano.py --smplh-fn reqs/smplh/tmp/male/model.npz --mano-left-fn reqs/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn reqs/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder reqs/smplh/

python scripts/merge_smplh_mano.py --smplh-fn reqs/smplh/tmp/female/model.npz --mano-left-fn reqs/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn reqs/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder reqs/smplh/

python scripts/merge_smplh_mano.py --smplh-fn reqs/smplh/tmp/neutral/model.npz --mano-left-fn reqs/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn reqs/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder reqs/smplh/

echo
echo "Done!"
echo
echo "Deleting tmp files"
rm -rf reqs/smplh/tmp/
echo 
echo "Done!"
