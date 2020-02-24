import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1-3dtP9wFNqBkzghovMtyhF6ckl1mD76s'
export_file_name = 'export.pkl'

classes = ['01 Real Estate Agent', '04 Financial Planner', '05 Economist', '02 Stock Broker', '06 Politician', '08 Entrepreneur', '07 Journalist', '03 Banker', '09 Divorce', '10 Corruption', '12 Critical Illness', '14 Illness', '11 Vacation', '13 Accident', '15 Baby', '16-Retrenchment', '17 Pay Raise', '20 Natural Disaster', '21 Interest Rate Rises', '30 Property Market Crash', '22 Interest Rate Falls', '23 Government Raises Tax', '29 Property Bubble', '25 Partisan Politics', '24 Government Lowers Tax', '26 Political Turmoil', '28 Stock Fever', '27 Debt Crisis', '31 Stock Market Crash', '36 Information Overload', '35 Land Rezoning', '34 Credit rating Downgrade', '18 Hot Money', '37 New Technology', '19 Government Bailout', '32 Inflation', '33 Stock Panic', '45 Property', '44 Stock', '40 Convertible Bond', '46 Bond', '39 Undeveloped Land', '41 Junk Bond', '43 Dividend Stock', '42 Growth Stock', '38 Hotel', '48 Insurance', '47 Trust Fund']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
