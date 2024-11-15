# Running

The vast majority of the program requires Python version 3.11 (there is one exception to this). It will not work with any other version.
To install all the required modules run `pip install -r requirements.txt` in the project root folder

To run the query interface program run `python main.py`

To process the dataset run the following files in order

- `refine_mesh.py`
- `fill_holes.py`
- `normals_check.py`
- `Normalisation.py`
- `normalisation_v2.py`

then run `feature_extraction.py`

If you want to collect the mesh data into a csv file you need to install python 3.9 with openmesh and use it to run `parse_meshes.py`

Run `knn_tsne.py` to see the tsne graph.

Run `accuracy.py` or `accuracy_knn.py` to calculate the corresponding precision and auc.
