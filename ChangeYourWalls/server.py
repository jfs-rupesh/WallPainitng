import argparse
import os
import base64
import xgboost
from mmseg.apis import init_model as init_segmentor, inference_model as inference_segmentor
import cv2
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestRegressor
# Load the model
with open('model_economy.pkl', 'rb') as file:
    model_e = pickle.load(file)
with open('model_royal.pkl', 'rb') as file:
    model_r = pickle.load(file)
with open('model_premium.pkl', 'rb') as file:
    model_p = pickle.load(file)



IMG_TYPE_SUPPORTED = ("jpg", "jpeg", "png", "webp")
CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
    'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
    'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
    'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
    'floor', 'flower', 'food', 'grass', 'ground', 'horse',
    'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
    'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
    'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
    'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
    'window', 'wood')

id_select = 55

try:
    model = init_segmentor(
        'models/pspnet_r101-d8_4xb4-80k_pascal-context-59-480x480.py', 
        'models/pspnet_r101-d8_480x480_80k_pascal_context_59_20210416_114418-fa6caaa2.pth', 
        device="cpu" if 1 else "cuda"
    )
    print("SEG MODEL LOADED")
except:
    model = None

def get_wall_mask_from_model(img):

    st = time.time()
    result = inference_segmentor(model, img)
    print("time", time.time()-st)

    res = np.array(
        result.pred_sem_seg.data[0], dtype=np.uint8
    )
    res[res != id_select] = 0
    res[res == id_select] = 1

    # cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
    # cv2.imshow("disp", res*255)
    # cv2.waitKey(0)

    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    print(res.shape)

    return res

def apply_pattern(img, pattern_img, mask_img=None, pts=None):

    if img is None or pattern_img is None:
        return None

    if mask_img is None:
        mask_img = get_wall_mask_from_model(img)

    H, W = pattern_img.shape[:2]
    pts1 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    og_h, og_w = img.shape[:2]
    if pts is None:
        pts2 = np.float32([[0, 0], [og_w, 0], [og_w, og_h], [0, og_h]])
    else:
        pass

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(pattern_img, matrix, img.shape[:2][::-1])
    
    result *= mask_img # pick only parts covering the selected class id (ex., bed, walls)
    img[result != 0] = img[result != 0]*0.2 + result[result != 0]*0.8

    return img

def error_in_function(err_message):
    return {"isSuccessful": False, "error": {"message": err_message}}

def get_numpy_image_from_string(
    string, 
    cv_imdecode=cv2.imdecode, 
    np_frombuffer=np.frombuffer, 
    np_uint8=np.uint8,
    base_64d=base64.b64decode,
    color=cv2.IMREAD_COLOR
):
    
    try:
        jpg_as_np = cv_imdecode(
            np_frombuffer(base_64d(string), dtype=np_uint8), color
        )
    except Exception as e:
        # print(e)
        return None
    else:
        return jpg_as_np

def numpy_image_to_b64(image):
    return base64.b64encode(
        cv2.imencode(".jpg", image)[1]
    ).decode()


def resize_to_desired_max_size_for_processing(img, max_size=1200):

    og_height, og_width = img.shape[:2]

    inter_ploation = (
        cv2.INTER_CUBIC
        if og_height < max_size and og_width < max_size
        else cv2.INTER_AREA
    )
    img_s_resized = (
        (int(og_width * max_size / og_height), max_size)
        if og_width < og_height
        else (max_size, int(og_height * max_size / og_width))
    )

    img = cv2.resize(img, img_s_resized, interpolation=inter_ploation)
    return img

pattern_dir = "demo/patterns"
ALL_PATTERNS_IMAGES = {
    idx:cv2.imread(f"{pattern_dir}/{f}")
    for idx, f in enumerate(os.listdir(pattern_dir), start=0)
    if f.lower().endswith(IMG_TYPE_SUPPORTED)
}

print(f"TOTAL PATTERNS LOADED  :: {len(ALL_PATTERNS_IMAGES)}")

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request

def get_api_server(api_prefix="", debug_mode=False):

    app = FastAPI(debug=debug_mode)

    app.add_middleware(
        CORSMiddleware, 
        allow_origins=["*"],
        allow_methods=["*"], 
        allow_headers=["*"],
        allow_credentials=True, 
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    if api_prefix:
        app_prefix = FastAPI(openapi_prefix=api_prefix)
        app.mount(api_prefix, app_prefix)
    else:
        app_prefix = None

    return app, app_prefix

## set fastapi server
app, app_prefix = get_api_server(api_prefix="/ai")

@app_prefix.get("/isReady/")
async def is_ready():
    return {"isReady": model is not None}

@app_prefix.post("/generateEstimate")
async def predict(request: Request):
    #Get the data from the POST request.
    #print("rupesh")
    data = await request.json()
    print(data)
    #Make prediction using model loaded from disk as per the data.
    #print(data['carpetArea'])
    transformData = transform(data)

    e_pred = model_e.predict([transformData])[0]
    r_pred = model_r.predict([transformData])[0]
    p_pred = model_p.predict([transformData])[0]
    if p_pred < e_pred:
        p_pred = e_pred * 1.291
    if r_pred < p_pred:
        r_pred = p_pred * 1.362
    prediction = {
        "ECONOMY": e_pred,
        "PREMIUM": p_pred,
        "ROYAL": r_pred
    }
    return prediction

@app_prefix.post("/apply")
async def main_api(request: Request):
    return api_call(await request.json(), _apply_pattern)

def _apply_pattern(img_b64, pattern_idx, mask_img_b64=None, pts=None):

    output_image = apply_pattern(
        get_numpy_image_from_string(img_b64), 
        ALL_PATTERNS_IMAGES[pattern_idx], 
        mask_img=get_numpy_image_from_string(mask_img_b64) if mask_img_b64 else None, 
        # pts=pts
    )
    return {
        "isSuccessful": True, 
        "body": {"outputImage":numpy_image_to_b64(output_image)}
    }

def api_call(data, func_to_call):

    if model is None:
        return error_in_function(
            "Server not loaded, please try again later"
        )

    try:
    # if 1:
        t1 = time.time()
        result = func_to_call(
            data.get("inputImage", None),
            data.get("patternId", 0),
            mask_img_b64=data.get("maskImage", None),
            pts=data.get("pts", None)
        )
        # print(result)
        result["totalExecutionTime"] = time.time() - t1
    except Exception as e:
        # print(e)
        message = "Exception at the server, " \
            "Report the issue to the admin, If it persists. "
        message += f"Give this message: ({str(e)})"
        
        result = error_in_function(message)
        
    return result

def test_each(src_dir):

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    for f in os.listdir(src_dir):
        if f.lower().endswith(IMG_TYPE_SUPPORTED):
            img = cv2.imread(f"{src_dir}/{f}")

            mask_img = get_wall_mask_from_model(img)

            for p_idx, p_img in ALL_PATTERNS_IMAGES.items():
                res_img = apply_pattern(img.copy(), p_img, mask_img=mask_img)
                
                cv2.imshow("disp", cv2.hconcat([img, res_img]))
                if cv2.waitKey(0) == ord("q"):
                    exit()

def transform(data):
    label_mapping_Property_type = {'INDEPENDENT_HOUSE': 0, 'APARTMENT': 1}
    label_mapping_BHK_type = {
        'RK1': 0,
        'BHK1': 1,
        'BHK2': 2,
        'BHK3': 3,
        'BHK4': 4,
        }

    input = []
    input.append(label_mapping_BHK_type[data["bhkType"]])
    input.append(data["carpetArea"])
    input.append(label_mapping_Property_type[data["houseType"]])
    input.append(data["isFirstTime"])
    input.append(data["isCeiling"])
    if "balcony" in data:
        input.append(data["balcony"])
    if "others" in data:
        if "dining" in data["others"]:
            input.append(1)
        else:
            input.append(0)
        if "passage" in data["others"]:
            input.append(1)
        else:
            input.append(0)
        if "utility" in data["others"]:
            input.append(1)
        else:
            input.append(0)
    else:
        input.append(0)
        input.append(0)
        input.append(0)
    return input

if __name__ == "__main__":

    og_img = cv2.imread("demo/source/test.jpg")
    og_img = resize_to_desired_max_size_for_processing(og_img, max_size=1600)

    b64 = numpy_image_to_b64(og_img)
    with open("img.txt", "w") as f:
        f.write(b64)
    

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port = 5001)