# zamorano
## a stremlit app to explore and review COCO datasets for instance segmentation tasks

This repo contains code for a streamlit app to review instance segmentation tasks.

## Run locally
Clone the repo and create a dedicate environment
```
$> git clone https://github.com/ciuffredaluca/zamorano.github
$> cd zamorano
$> conda create -n env python=3.10
$> conda activate zamorano
$> pip install -r requirements.txt
```

Run the app
```
$> streamlit run main.py
```

## Run from Docker container

```
$> docker build -t zamorano .
$> docker run -p 8501:8501 zamorano
```