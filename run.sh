# Create environment
conda create -n mlops-cloud-deploy "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge

# Use created environment
conda activate mlops-cloud-deploy