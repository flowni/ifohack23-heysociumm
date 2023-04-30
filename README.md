# DLR Challenge - ifoHack 2023

## Inspiration
We started the hackathon as a team of four - two techies and two sociologists. After a full day of setup, our sociologist couldn't even take a look at the spatial data to bring in their knowledge about social influences of land prices, which is what we needed to build a prediction model around. With frustration about the painful setup, they gave up. 

To make data exploration easier for our fallen friends, we made it our mission to build a data literacy dashboard so that anyone can explore spatial data without much setup.

## What it does
Our dashboard will show you every insight you want to know about the land prices in Bremen. From basic questions like 'How is the density of young people in each neighborhood?' to more advanced questions like 'How much does the distance from the center contribute to the land price in the neighborhood Altbremen?', you can answer them all.

## How we built it
We built it using the provided data together with Streamlit, XGBoost and Shapley values.

## Challenges we ran into
Losing two of our teammates and all the setup that the spatial data needed (a lot of data, new programs and libraries).

## Accomplishments that we're proud of
Getting everything to work in such a short time and learning all these new geospatial tools, as well as streamlit!

Thanks for organizing the hackathon - Nico and Morris
## Setup

Install all dependencies in conda env

## Run

To run the streamlit web app, run

```bash
streamlit run main.py
```

To run the model evaluation, go to folder challenge_evaluation

```bash
cd challenge_evaluation
```

```bash
python model.py
```

## Authors

- Morris Kurz
- Nicholas Wolf
