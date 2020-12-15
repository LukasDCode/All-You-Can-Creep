# export the environment

conda env export > environment.yml

# import the environment

conda env create -f environment.yml