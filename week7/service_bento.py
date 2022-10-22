import asyncio

import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class UserProfile(BaseModel):
    name: float
    age: float
    country: float
    rating: float


model_ref = bentoml.sklearn.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5')
model_ref2 = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')

model_runner = model_ref2.to_runner()
#model_runner2 = model_ref.to_runner()

svc = bentoml.Service('mlzoomcamp_homework', runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)

    return prediction
