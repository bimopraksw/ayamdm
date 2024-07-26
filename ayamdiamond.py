import os
import requests
import time
import json
from colorama import init, Fore, Style
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import cv2
from io import BytesIO

init(autoreset=True)

def get_random_color():
    colors = [Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
    return random.choice(colors)

with open('ayamquery.txt', 'r') as file:
    lines = file.readlines()

authorizations = [line.strip() for line in lines]

headers_template = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'authorization': '',
    'cache-control': 'no-cache',
    'content-length': '0',
    'content-type': 'application/octet-stream',
    'origin': 'https://game.chickcoop.io',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://game.chickcoop.io/',
    'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
}

task_url = "https://api.chickcoop.io/mission/task"
claim_url = "https://api.chickcoop.io/mission/task/claim"
state_url = "https://api.chickcoop.io/mission/task/social"
ambassador_url = "https://api.chickcoop.io/mission/ambassador"
ambassador_state = "https://api.chickcoop.io/mission/ambassador/complete"
ambassador_claim = "https://api.chickcoop.io/mission/ambassador/claim"

def load_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content))
    return img

def extract_grid(image, grid_size=3, target_size=(100, 100)):
    width, height = image.size
    cell_width = width // grid_size
    cell_height = height // grid_size
    grid_images = []

    for i in range(grid_size):
        for j in range(grid_size):
            left = j * cell_width
            top = i * cell_height
            right = (j + 1) * cell_width
            bottom = (i + 1) * cell_height
            cell_img = image.crop((left, top, right, bottom))
            cell_img_resized = cell_img.resize(target_size, Image.LANCZOS)
            grid_images.append(cell_img_resized)

    return grid_images

def draw_grid(image, grid_size=3):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    cell_width = width // grid_size
    cell_height = height // grid_size

    for i in range(grid_size + 1):
        draw.line([(i * cell_width, 0), (i * cell_width, height)], fill="red", width=2)
    for j in range(grid_size + 1):
        draw.line([(0, j * cell_height), (width, j * cell_height)], fill="red", width=2)

    return image

def compare_images(img1, img2):
    img1_gray = np.array(img1.convert('L'))
    img2_gray = np.array(img2.convert('L'))

    ssim_score, _ = ssim(img1_gray, img2_gray, full=True)
    mse_score = mse(img1_gray, img2_gray)

    img1_hist = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    hist_dist = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)

    return ssim_score, mse_score, hist_dist

def match_eggs(grid_image_url, egg_image_urls, target_size=(100, 100)):
    large_image = load_image(grid_image_url)

    grid_image_with_lines = draw_grid(large_image.copy())
    grid_image_with_lines.save("large_image_with_grid.png")
    
    grid_images = extract_grid(large_image, target_size=target_size)

    egg_images = [load_image(url).resize(target_size, Image.LANCZOS) for url in egg_image_urls]
    
    positions = [-1] * len(egg_images)
    used_positions = set()

    for egg_idx, egg_img in enumerate(egg_images):
        best_ssim_score = -1
        best_mse_score = float('inf')
        best_hist_dist = -float('inf')
        best_position = -1
        
        for idx, grid_img in enumerate(grid_images):
            if idx in used_positions:
                continue
            
            ssim_score, mse_score, hist_dist = compare_images(egg_img, grid_img)
            
            if (hist_dist > best_hist_dist or
                (hist_dist == best_hist_dist and ssim_score > best_ssim_score) or
                (hist_dist == best_hist_dist and ssim_score == best_ssim_score and mse_score < best_mse_score)):
                best_ssim_score = ssim_score
                best_mse_score = mse_score
                best_hist_dist = hist_dist
                best_position = idx
        
        if best_position != -1:
            positions[egg_idx] = best_position
            used_positions.add(best_position)
    
    return positions

def get_challenge_data(auth):
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': auth,
        'origin': 'https://game.chickcoop.io',
        'priority': 'u=1, i',
        'referer': 'https://game.chickcoop.io/',
        'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    }

    response = requests.get('https://api.chickcoop.io/user/challenge', headers=headers)
    return response.json()

def check_free_spin(headers):
    wheel_url = "https://api.chickcoop.io/v2/wheel"
    claim_spin = "https://api.chickcoop.io/wheel/claim"
    requests.post(claim_spin, headers=headers)
    response = requests.get(wheel_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        wheel_state = data['data']['wheelState']
        next_free_spin = data['data']['wheelState']['nextTimeFreeSpin']

        if wheel_state['isAvailableFreeSpin']:
            return True
        elif next_free_spin is None or next_free_spin < datetime.now().timestamp():
            return True
    return False

def spin_wheel(headers):
    spin_url = "https://api.chickcoop.io/v2/wheel/spin"
    payload = {
        "mode": "free"
    }
    response = requests.post(spin_url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if data.get('ok'):
            return True
    return False

def complete_and_claim_tasks(auth):
    headers = headers_template.copy()
    headers['authorization'] = auth
    
    # Fetch tasks
    responseTask = requests.get(task_url, headers=headers)
    responseAmbassador = requests.get(ambassador_url, headers=headers)
    
    responseTask = responseTask.json()
    responseAmbassador = responseAmbassador.json()
    if responseTask['ok']:
        for task in responseTask['data']['social']:
            payload = { 
                "task": {
                    "id": task['id'],
                    "name": task['name'],
                    "check": task['check'],
                    "url": task['url'],
                    "gemsReward": task['gemsReward'],
                    "achieved": task["achieved"],
                    "rewarded": task['rewarded']
                }
            }
            requests.post(state_url, headers=headers, json=payload)
            payload["task"]["achieved"] = True
            requests.post(claim_url, headers=headers, json=payload)
        for task in responseTask['data']['daily']:
            payloadcheckin = { 
                "task": {
                    "id": task['id'],
                    "name": task['name'],
                    "gemsReward": task['gemsReward'],
                    "achieved": task["achieved"],
                    "rewarded": task['rewarded']
                }
            }
            requests.post(state_url, headers=headers, json=payloadcheckin)
            payload["task"]["achieved"] = True
            requests.post(claim_url, headers=headers, json=payloadcheckin)
    
    if responseAmbassador['ok']:
        for country in responseAmbassador['data']:
            for task in country['tasks']:
                payload = {
                    "task" : {
                        "url": task['url'],
                        "name": task['name'],
                        "image": task['image'],
                        "subscribers": task['subscribers'],
                        "gemsReward": task['gemsReward'],
                        "achieved": False,
                        "rewarded": False
                    }
                }
                requests.post(ambassador_state, headers=headers, json=payload)
                payload["task"]["achieved"] = True
                requests.post(ambassador_claim, headers=headers, json=payload)

def claim_gift(auth):
    headers = headers_template.copy()
    headers['authorization'] = auth
    response = requests.post('https://api.chickcoop.io/gift/claim', headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

def verify_captcha(auth, sequence):
    headers = headers_template.copy()
    headers['authorization'] = auth
    verify_url = 'https://api.chickcoop.io/user/challenge/verify'
    payload = {
        "sequence": sequence
    }
    response = requests.post(verify_url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('ok', False)
    return False

previous_results = {}

output_lines = [""] * len(authorizations)

def fetch_and_print_user_data(auth, index, initial_run):
    headers = headers_template.copy()
    headers['authorization'] = auth
    
    if initial_run:
        complete_and_claim_tasks(auth)
    
    while True:
        try:
            time.sleep(2)

            gift_response = claim_gift(auth)
            if gift_response:
                if gift_response.get('ok'):
                    chest_count = previous_results.get(index, {}).get('chest_count', 0) + 1
                    previous_results[index] = {'chest_count': chest_count}
                else:
                    error_message = gift_response.get('error')
                    if error_message == "Unavailable Chest":
                        chest_count = previous_results.get(index, {}).get('chest_count', 0)
                    elif error_message == "Need verify not a bot":
                        challenge_data = get_challenge_data(auth)
                        main_image_url = challenge_data['data']['challenge']['mainImage']
                        hint_image_urls = challenge_data['data']['challenge']['hintImages']
                        sequence = match_eggs(main_image_url, hint_image_urls)

                        if verify_captcha(auth, sequence):
                            chest_count = previous_results.get(index, {}).get('chest_count', 0)
                        else:
                            chest_count = previous_results.get(index, {}).get('chest_count', 0)
                            print("CAPTCHA verification failed. Handle as needed.")

                    else:
                        # Handle other errors or unexpected cases
                        print(f"Unexpected error: {error_message}.")
                        chest_count = previous_results.get(index, {}).get('chest_count', 0)

                output_lines[index] = f"Akun {index + 1} - Gems: {gift_response.get('data', {}).get('gem', 'Unknown')} | Chest Count: {chest_count}"
                print_output()

            if check_free_spin(headers):
                spin_wheel(headers)
          
        except Exception as e:
            output_lines[index] = Fore.RED + f"Error fetching data for Akun {index + 1}: {e}"
            print_output()
            time.sleep(5)  # Wait before retrying

def print_output():
    # Clear the terminal
    print("\033c", end="")  # ANSI escape code to clear the screen
    # Print all results at once
    print("\n".join(output_lines), end="\r", flush=True)

# Initial run flag
initial_run = True

try:
    while True:
        results = []
        num_workers = len(authorizations)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(fetch_and_print_user_data, auth, index, initial_run) for index, auth in enumerate(authorizations)]
            for future in futures:
                result = future.result()  # Wait for all threads to complete
                if result:
                    results.append(result)
        
        initial_run = False

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Exiting...")
