# !/usr/bin/env python3
## DOCS https://huggingface.co/SnypzZz/Llama2-13b-Language-translate
## Translate not fully tested still experimental

## BASED https://github.com/xtekky/gpt4free/blob/main/docs/providers-and-models.md

import re, time, random, requests, json, csv, asyncio, os, subprocess
import concurrent.futures
import g4f
from g4f.client import Client as G4FClient
from datetime import datetime, timedelta
from lxml import html
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from wordpress_xmlrpc import Client as WPClient, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from playwright.sync_api import sync_playwright
from colorama import init, Fore, Style
import collections
collections.Iterable = collections.abc.Iterable
import warnings
warnings.filterwarnings("ignore")
init()

keywords = "travel.txt"

### ROLE WAJIB ADMINISTRATOR ###
login = [
    "admin|Sukses@123",
    "admin|Sukses@123"
]

DOMAIN = "juzgamer.tech"
LIST_CATEGORY = ["Travel Info", "Travel Tips", "Travel Guide"]

SPIN_LOCAL = False
SPIN_QB = False
GOTRANSLATE = False     # Not Tested
CSV = False
DEBUG = True

IMG_GOOGLE = True
IMG_BING = False
IMG_CDN = True
MULTI_IMAGE = False

### CONFIG QUILLBOT PREMIUM ###
QB_ID = ""
QB_PW = ""

### CONFIG SCHEDULE OR BACKDATE ###
SLEEP_INTERVAL = 28800   # 8 jam
# SLEEP_INTERVAL = 86400   # 1 hari
SCHEDULE = False
BACKDATE = False
TIME_START = "2025-02-09 20:02:00"      # Y:M:D H:M:S mode 24H

### CONFIG AI ###
AIO_AI = True
OLLAMA = "127.0.0.1:8080"
MAXRETRY = 3

### CONFIG PROXY FOR AIO AI ###
proxy_resident = False
if proxy_resident:
    PROXY_SERVER = "http://127.0.0.1:801"       # WAJIB residential rotating proxy mahal untuk produksi
else:
    SCRAPE_PROXY = False    # if true use url_proxy else use local file proxy.txt   WAJIB UBUNTU
    URL_PROXY = "https://raw.githubusercontent.com/r00tee/Proxy-List/refs/heads/main/Socks5.txt"
    TYPE_PROXY = "socks5://"
    PROXY_SERVER = "http://127.0.0.1:7539"      # proxy list to pproxy http

JOB = None  # Dont touch

def generate_content(keyword, images):
    global JOB
    print(Fore.YELLOW + '> Generate Title' + Style.RESET_ALL)
    JOB = "Title"
    prompt = f"Write SEO title limited to 7 words for keyword '{keyword}', just reply in one line and skip any special characters only letters and numbers are allowed."
    
    if AIO_AI:
        title = asyncio.run(gpt4free(prompt))
    else:
        title = ollama_ai(prompt)

    if GOTRANSLATE:
        title = translate_content(title)
    title = capitalize(title)
    if DEBUG:
        print(f"Judul : {title}\n")
 
    print(Fore.YELLOW + '> Generate Deskripsi' + Style.RESET_ALL)
    JOB = "Desc"
    prompt = f"Craft an SEO description for the article limited to 22 words, and retain the original key phrase {title} in the meta description. Only letters and numbers are allowed without Here are bla bla etc."
    
    if AIO_AI:
        desc = asyncio.run(gpt4free(prompt))
    else:
        desc = ollama_ai(prompt)
        
    if SPIN_QB:
        desc = spin_quillbot(desc)
    if SPIN_LOCAL:
        desc = spin_content(desc)
    if GOTRANSLATE:
        desc = translate_content(desc)
    if DEBUG:
        print(f"Deskripsi : {desc}\n")

    print(Fore.YELLOW + '> Generate Tags' + Style.RESET_ALL)
    JOB = "Tags"
    prompt = f"Generate SEO related keyword for {keyword} and separate by comma , Without Here are bla bla etc. Max 5 output"
    
    if AIO_AI:
        related = asyncio.run(gpt4free(prompt))
    else:
        related = ollama_ai(prompt)
        
    if GOTRANSLATE:
        related = translate_content(related)
    if DEBUG:
        print(related)
    
    print(Fore.YELLOW + '> Generate Topik' + Style.RESET_ALL)
    JOB = "Topic"
    topics = []
    seen = set()
    prompt = f"Write me topic for keyword '{keyword}' each topic in single line dont use new sub and begins with >. Without Here are bla bla etc. Max topic output is 8 result"

    if AIO_AI:
        topic = asyncio.run(gpt4free(prompt))
    else:
        topic = ollama_ai(prompt)
        
    if DEBUG:
        print(f"{topic}\n")

    if '\n' in topic:
        for line in topic.strip().split('\n'):
            output = line.replace('> ', '').replace('*', '').replace(':', '').replace('>', '')
            cleaned_line = replace_text(output)
            
            if len(cleaned_line) > 2 and cleaned_line not in seen:
                topics.append(cleaned_line.strip())
                seen.add(cleaned_line.strip())

    print(Fore.YELLOW + '> Generate Paragraf' + Style.RESET_ALL)
    JOB = "Sentence"
    list_topic = []
    cp = 1
    count_img = 0
    for topic in topics:
        get_img = images[count_img]
        img_url, img_title = get_img.split('@')

        paragraf = None
        prompt = f"""
            I Want You To Act As A Content Writer Who Is A Very Proficient SEO Writer And Writes In English Fluently.

            Task:
            - Produce an 250-word paragraph on the topic '{topic}'.
            - Use appropriate headings and sub-headings related to '{topic}'.
            - Adding value to this paragraph.
            - Follow all google adsense content guidelane.
            - Follow google E-E-A-T

            Requirements:
            - Write a 100% unique, creative, and human-like paragraphe. Write The paragraph In Your Own Words Rather Than Copying And Pasting From Other Sources. 
            - Write In A Conversational Style As Written By A Human (Use An Informal Tone, Utilize Personal Pronouns, Keep It Simple, Engage The Reader, Use The Active Voice, Keep It Brief, Use Rhetorical Questions, and Incorporate Analogies And Metaphors). 
            - Consider perplexity and burstiness when creating content, ensuring high levels of both without losing specificity or context. 
            - Utilize contractions, idioms, transitional phrases, interjections, and colloquialisms.
            - Use fully detailed paragraphs that engage the reader.
            - Avoid repetitive phrases and unnatural sentence structures.
            - Avoid Introduction, FAQs and Conclusion.
            - Avoid any duplicated sentence.
        """
        
        if AIO_AI:
            paragraf = asyncio.run(gpt4free(prompt))
        else:
            paragraf = ollama_ai(prompt)

        if SPIN_QB:
            paragraf = spin_quillbot(paragraf)
        if SPIN_LOCAL:
            paragraf = spin_content(paragraf)
        if GOTRANSLATE:
            paragraf = translate_content(paragraf)

        if len(paragraf.split()) > 100:
            paragraf = split_paragraph(paragraf)

        if MULTI_IMAGE:
            list_topic.append(f"<br><center><img style='width: 70%;' src='{img_url}' alt='{img_title}' title='{img_title}'><br>Illustration image by Pinterest</center><br><h2>{topic}</h2><p>{paragraf}</p>")
            count_img += 1
        else:
            list_topic.append(f"<h2>{topic}</h2><p>{paragraf}</p>")
        if DEBUG:
            print(f"Generated Paragraf {cp} - {topic}\n{paragraf}")
            print()
            cp += 1

    full_paragraf = " ".join(list_topic)
    content_paragraf = ", ".join(topics)

    print(Fore.YELLOW + '> Generate FAQs' + Style.RESET_ALL)
    JOB = "Faq"
    prompt = f"Please generate a FAQ section in valid HTML format. Each FAQ should include a question start with Q : and a concise answer start with A : Use <h3>, and <p> <li> tags for structure without <div> <a> <class>. Make sure the HTML is clean and easy to integrate. FAQs related to '{content_paragraf}'. Avoid text 'Here is the FAQ section in valid HTML format:' or 'Note: You' or similar like it"
        
    if AIO_AI:
        faq = asyncio.run(gpt4free(prompt))
    else:
        faq = ollama_ai(prompt)

    if GOTRANSLATE:
        faq = translate_content(faq)
    if DEBUG:
        print(faq)

    print(Fore.YELLOW + '> Generate Conclusion' + Style.RESET_ALL)
    JOB = "Conclusion"
    prompt = f"Write SEO conclusion paragraph about {content_paragraf}"

    if AIO_AI:
        conclusion = asyncio.run(gpt4free(prompt))
    else:
        conclusion = ollama_ai(prompt)

    if SPIN_QB:
        conclusion = spin_quillbot(conclusion)
    if SPIN_LOCAL:
        conclusion = spin_content(conclusion)
    if GOTRANSLATE:
        conclusion = translate_content(conclusion)
    if DEBUG:
        print(conclusion)
 
    if MULTI_IMAGE or not images:
        content = (f"<p>{desc}</p>{full_paragraf}{faq}<p>{conclusion}</p>")
    else:
        random_img = random.choice(images)
        img_url, img_title = random_img.split('@')
        content = (f"<p>{desc}</p><center><img style='width: 70%;' src='{img_url}' alt='{img_title}' title='{img_title}'><br>Illustration image by Pinterest</center><br>{full_paragraf}{faq}<p>{conclusion}</p>")

    content = minify_text(content)
    random_category = random.choice(LIST_CATEGORY)

    ALL_DATA.append({
        'title': title,
        'content': content,
        'cat': random_category,
        'tag': related,
        'img': img_url
    })

    with open('debugbykw.html', 'w') as file:
        file.write(content)

async def gpt4free(prompt):
    if JOB == "Faq":
        working_list = [
            "NexraChatGPT:gpt-4",
            "Cloudflare:llama-3.2-11b",
            "Airforce:llama-3.2-90b"
        ]
    else:
        working_list = [
            "GizAI:gpt-4o",
            "DarkAI:gpt-4o",
            "Pizzagpt:gpt-4o-mini",
            "DeepInfraChat:llama-3.1-8B",
            "TeachAnything:llama-3.1-70b",
            "HuggingChat:llama-3.2-11b",
            "AIUncensored:ai_uncensored",
            "Upstage:solar-pro"
        ]

    count = 0
    while True:
        try:
            random.shuffle(working_list)
            get_working = random.choice(working_list)
            get_provider, get_model = get_working.split(':')
            if JOB != "Sentence":
                print(Fore.GREEN + f"* {get_provider}" + Style.RESET_ALL)
            providers = getattr(g4f.Provider, get_provider)
            client_gpt = G4FClient(proxies=PROXY_SERVER)
            response = await client_gpt.chat.completions.async_create(
                model=get_model,
                provider=providers,
                messages=[
                    {"role": "system", "content": "You must Provide the answer directly without additional explanation or additional information from you with tone informational. make sure use whitespace each word."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.8,
                max_tokens=8192,
            )
            output = response.choices[0].message.content
            
            if "404" not in output or "403" not in output or "500" not in output:
                if ' ' in output:
                    if JOB == "Title":
                        output1 = replace_text(output)
                        title  = clean_title(output1)
                        is_badword = check_badwords(title)
                        if not is_badword:
                            if len(title.split()) > 5:
                                return title
                    if JOB == "Desc":
                        desc = replace_text(output)
                        desc = desc.replace('"',"")
                        if len(desc.split()) > 8:
                            return desc
                    if JOB == "Topic":
                        if len(output.split()) > 10:
                            return output
                    if JOB == "Sentence":
                        if 'lt;' not in output and '**' not in output and '##' not in output and '">' not in output:
                            paragraf = replace_text(output)
                            paragraf = paragraf.replace("<h2>","").replace("</h2>","").replace("<h3>","").replace("</h3>","")
                            paragraf = paragraf.replace("\n","</p><p>")
                            paragraf = minify_text(paragraf)
                            crot = f"* {get_provider} - {len(paragraf.split())}"
                            print(Fore.GREEN + crot + Style.RESET_ALL)
                            if len(paragraf.split()) > 100:
                                if len(paragraf.split()) < 700:
                                    return output
                                else:
                                    print(Fore.RED + f"! {get_provider} - Words over detected" + Style.RESET_ALL)
                            else:
                                print(Fore.RED + f"! {get_provider} - Words less detected" + Style.RESET_ALL)
                    if JOB == "Tags":
                        related = replace_text(output)
                        related = minify_text(related)
                        related = related.replace('"','').replace("\n","")
                        if len(output.split()) > 8:
                            return related
                    if JOB == "Faq":
                        if 'lt;' not in output and '**' not in output and '##' not in output and '">' not in output:
                            if len(output.split()) > 10:
                                faq = replace_text(output)
                                return faq
                    if JOB == "Conclusion":
                        if 'Note:' not in output and 'HTML' not in output:
                            if len(output.split()) > 10:
                                conclusion = replace_text(output)
                                return conclusion
                else:
                    print(Fore.RED + f"! No Space Detected - {get_provider}" + Style.RESET_ALL)
                    get_proxies()
                    time.sleep(5)

        except Exception:
            if count == MAXRETRY:
                if proxy_resident:
                    print("Reached Max Retry Long Sleep ON")
                    time.sleep(120)
                else:
                    get_proxies()
                    time.sleep(5)
                count = 0
            else:
                time.sleep(5)
                count += 1

def replace_text(input_string):
    replacements = {
        "|endoftext|": "",
        "||im_end] |": "",
        "]|im_end|>": "",
        "<|im_end|": "",
        "im_end>": "",
        "|eot_id|>[start_header_Id]>>assistantend_ header-iD|horr": "",
        ".|eot_id|>[start_header_Id]>>assistant/end_ header-iD|horr=>": "",
        "```html": "",
        "```": "",
        "***": "",
        "**": "",
        "###": "",
        "##": "",
        ">>.": "",
        ">>": "",
        "Im_end>": "",
        "im_end": "",
        "|END |": "",
        "|end|": "",
        "|": ""
    }

    for i in range(3):
        for old, new in replacements.items():
            input_string = input_string.replace(old, new)
    return input_string

def ollama_ai(prompt):
    url =f"http://{OLLAMA}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.2:latest",
        "api_key": "ollama",
        "messages": [
            {"role": "system", "content": "You must Provide the answer directly without additional explanation or additional information from you with tone informational."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,
        "max_tokens": -1,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=1200)
    parsed_data = json.loads(response.text)
    content_value = parsed_data['choices'][0]['message']['content']
    time.sleep(3)
    return content_value

def get_proxies():
    if SCRAPE_PROXY:
        list_proxy = []
        while True:
            try:
                response = requests.get(URL_PROXY)
                response.raise_for_status()
                gproxies = response.text.strip().splitlines()
                list_proxy = list(set(gproxies))
                ip = random.choice(list_proxy)
                break
            except Exception as e:
                print(f"FAILED GET PROXY: {e}")
    else:
        while True:
            try:
                with open('proxy.txt', 'r') as file:
                    lines = file.readlines()
                if not lines:
                    print("File Proxy Empty.")
                    time.sleep(300)
                else:
                    ip = random.choice(lines).strip()
                    break
            except FileNotFoundError:
                print("The file 'proxy.txt' was not found.")
                time.sleep(300)
    
    if '//' not in ip:
        ip = TYPE_PROXY + ip
    kill_command = ['sudo', 'screen', '-x', 'proxy', '-x', '-X', 'kill']
    start_command = ['sudo', 'screen', '-Sdm', 'proxy', 'pproxy', '-v', '-l', 'http://:7539', '-r', ip]
    subprocess.run(kill_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(start_command)
    time.sleep(2)

    return True

def capitalize(text):
    words = text.split()
    if words:
        words[0] = words[0].capitalize()
        words[1:] = [word.lower() for word in words[1:]]
    
    return ' '.join(words)

def translate_content(input_text):
    model = MBartForConditionalGeneration.from_pretrained("llama/translate")
    tokenizer = MBart50TokenizerFast.from_pretrained("llama/translate", src_lang="en_XX")
    model_inputs = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )
    hasil = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    time.sleep(3)
    return hasil

def spin_quillbot(input_text):
    sentences = re.split(r'(?<=[.!?]) +', input_text)
    max_workers = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(quillbot, sentence): i for i, sentence in enumerate(sentences)}
        spun_sentences = [None] * len(sentences)
        
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                spun_sentences[index] = future.result()
            except Exception as e:
                print(f"Error processing sentence {index}: {e}")
                spun_sentences[index] = sentences[index]

    executor.shutdown(wait=True)
    spun_paragraph = ' '.join(spun_sentences)
    return spun_paragraph

def spin_content(input_text):
    sentences = re.split(r'(?<=[.!?]) +', input_text)
    # sentences = re.split(r'\.\s*', input_text)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained('./llama/text-rewriter')
    bot = AutoModelForSeq2SeqLM.from_pretrained('./llama/text-rewriter').to(device)
    article = ''
    for sentence in sentences:
        sentence = sentence.replace('- ', '').replace('-- ', '').replace(':', '').replace(';', '')
        if sentence:
            if DEBUG:
                print(f"Sentence ORI = {sentence}\n")
            input_ids = tokenizer(f'paraphraser: {sentence}', return_tensors='pt', padding='longest', truncation=True, max_length=128).input_ids
            output = bot.generate(
                input_ids,
                num_beams=5,
                num_beam_groups=5,
                num_return_sequences=5,
                repetition_penalty=10.0,
                diversity_penalty=3.0,
                no_repeat_ngram_size=2,
                max_length=128
            )
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            random_sentence = random.choice(decoded_output)
            random_sentence = random_sentence.replace('- ', '').replace('-- ', '').replace(':', '').replace(';', '').replace('Paraphraser', '').replace('paraphraser', '').replace('..', '.')
            if DEBUG:
                print(f"Sentence Spin = {random_sentence}\n")
            result = f" {random_sentence}"
            article += result
    return article

def qb_login():
    url_login = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCKK18QdZG32zJeaAJ8awVpCRKgIATUtTE"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.8",
        "content-type": "application/json",
        "origin": "https://quillbot.com",
        "qb-dialect": "en-us",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    }

    data = {
        "returnSecureToken": True,
        "email": QB_ID,
        "password": QB_PW
    }

    response = requests.post(url_login, headers=headers, json=data)
    if response.status_code == 200:
        response_dict = response.json()
        id_token = response_dict.get('idToken')
        return id_token
    else:
        return None
    
def qb_logout():
    global tokenqb
    url_logout = "https://quillbot.com/api/auth/sign-out-from-all"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Host": "quillbot.com",
        "platform-type": "webapp",
        "qb-product": "SETTINGS",
        "Referer": "https://quillbot.com/settings?menu=sessions",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "useridtoken": tokenqb,
        "webapp-version": "15.145.3"
    }

    response = requests.get(url_logout, headers=headers)
    return response

def quillbot(input_text):
    global tokenqb
    url_spin = "https://api.quillbot.com/api/paraphraser/single-paraphrase/8"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.8",
        "content-type": "application/json",
        "origin": "https://quillbot.com",
        "priority": "u=1, i",
        "qb-dialect": "en-us",
        "qb-product": "PARAPHRASER",
        "referer": "https://quillbot.com/paraphrasing-tool",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "useridtoken": tokenqb
    }

    data = {
        "fthresh": -1,
        "autoflip": False,
        "wikify": False,
        "inputLang": "en",
        "strength": 8,
        "quoteIndex": -1,
        "text": input_text,
        "frozenWords": [],
        "nBeams": 4,
        "freezeQuotes": True,
        "preferActive": False,
        "dialect": "US",
        "promptVersion": "v3",
        "multilingualModelVersion": "v2"
    }

    while True:
        response = requests.post(url_spin, headers=headers, json=data)
        if response.status_code == 200:
            spinku = response.json()
            getspin = random.choice(spinku['data'][0]['paras_9'])['alt']
            return getspin
        else:
            print(Fore.RED + f"! QB Error Spin Check Auth" + Style.RESET_ALL)
            return input_text
def split_paragraph(paragraph):
    sentences = paragraph.split('. ')
    mid_index = len(sentences) // 2
    first_half = '. '.join(sentences[:mid_index + 1])
    second_half = '. '.join(sentences[mid_index + 1:])
    return f"<p>{first_half.strip()}.</p> <p>{second_half.strip()}</p>" if second_half else f"<p>{first_half.strip()}.</p>"

def minify_text(html_content):
        html_content = re.sub(r'\s+', ' ', html_content)
        html_content = re.sub(r'>\s+<', '><', html_content)
        html_content = re.sub(r'>\s+', '>', html_content)
        html_content = re.sub(r'\s+<', '<', html_content)
        return html_content.strip()

def process_first_line(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        time.sleep(2)
    return first_line

def delete_first_line(filename):
    with open(filename, 'r') as file:
        remaining_lines = file.readlines()[1:]
    with open(filename, 'w') as file:
        file.writelines(remaining_lines)

def count_lines_in_file():
    global keywords
    with open(keywords, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
    print(f"Remaining Keywords {line_count}")

def clean_title(title):
    title = re.sub(r'[`~!@#$%^&*()_\-+=\[\]{};:\'"\\|\/,.<>?]', '', title)
    title = re.sub(r'\s\s+', ' ', title)
    title = title.strip()
    return title

def is_valid_title(title):
    return bool(re.match(r'^[\x20-\x7E]*$', title))

def check_variable(var):
    if var is None:
        print("Variable is None")
    elif isinstance(var, str):
        print("Variable is a string")
    elif isinstance(var, list):
        print("Variable is a list")
    else:
        print(f"Variable is of type: {type(var)}")

def bing_images(keyword):
    if DEBUG:
        print("Playwright Scrape Bing Images")
    keyword = keyword.replace('how to ', '') 
    kwplus = keyword.replace(' ', '+')
    url = f'https://www.bing.com/images/search?q={kwplus}+site:pinterest.com&first=1'
    count = 1
    while True:
        try:
            with sync_playwright() as p:
                browser = p.webkit.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Linux; Android 4.4.2; P1 Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Crosswalk/23.53.589.4 Safari/537.36',
                    ignore_https_errors=True
                )
                page = context.new_page()
                page.goto(url, timeout=60000)
                for i in range(3):
                    page.evaluate("window.scrollBy(0, 1000)")
                    time.sleep(2)
                time.sleep(5)
                content = page.content()
                browser.close()
        except Exception as e:
            if count == 2:
                print("Reached maximum retry bing images scrape")
                break
            else:
                print(f"Attempt {count} failed. Retrying...")
                count += 1

        tree = html.fromstring(content)
        img_elements = tree.xpath("//div[@class='imgpt']/a")
        list_img = []
        for img in img_elements:
            img_raw = img.get('m', '')
            data_img = json.loads(img_raw)
            img_url = data_img['murl']
            img_title = data_img['t']

            img_title = clean_title(img_title)

            if img_url and img_title and "pin" not in img_title.lower() and is_valid_title(img_title):
                if IMG_CDN:
                    img_url = cdn_images(img_url)
                try:
                    response = requests.get(img_url)
                    response.raise_for_status()

                    if response.status_code == 200:
                        list_img.append(f"{img_url}@{img_title}")
                    else:
                        continue
                except Exception:
                    continue
        return list_img

def google_images(keyword):
    if DEBUG:
        print("Scrape Google Images")
    list_img = []
    keyword = keyword.replace('how to ', '') 
    kwplus = keyword.replace(' ', '+')
    url = f'https://www.google.com/search?q={kwplus}+site:pinterest.com&source=lnms&tbm=isch'
    cek_total = True
    while True:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 4.4.2; P1 Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Crosswalk/23.53.589.4 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        img_elements = tree.xpath("//div[@class='islrtb isv-r']")

        for img in img_elements:
            img_url = img.get('data-ou', '')
            img_title = img.get('data-pt', '')

            img_title = clean_title(img_title)

            if img_url and img_title and "pin" not in img_title.lower() and is_valid_title(img_title):
                if IMG_CDN:
                    img_url = cdn_images(img_url)
                try:
                    response = requests.get(img_url)
                    response.raise_for_status()

                    if response.status_code == 200:
                        list_img.append(f"{img_url}@{img_title}")
                    else:
                        continue
                except Exception:
                    continue
        
        if cek_total:
            if len(list_img) < 10:
                url = f'https://www.google.com/search?q={kwplus}&source=lnms&tbm=isch'
                cek_total = False
            else:
                break
        else:
            break
    if len(list_img) < 10:
        return None
    else:
        list_img = [img for img in list_img if "fbsbx.com" not in img]
        return list_img

def cdn_images(url):
    if url.startswith('http://'):
        return 'https://wsrv.nl/?url=' + url[len('http://'):]
    elif url.startswith('https://'):
        return 'https://wsrv.nl/?url=' + url[len('https://'):]
    return url

def update_post(title, tag_variable, content, cat, img):
    global scheduled_date, id, pw
    tags = [tag.strip() for tag in tag_variable.split(',')]
    WP = WPClient(f"https://{DOMAIN}/xmlrpc.php", id, pw)

    if SCHEDULE:
        time_post = 'future'
    else:
        time_post = 'publish'

    try:
        post = WordPressPost()
        post.title = title
        post.slug = title
        post.content = content
        post.terms_names = {
            'post_tag': tags,
            'category': [cat]
        }
        post.custom_fields = [
            {'key': 'fifu_image_url', 'value': img},
            {'key': 'fifu_image_alt', 'value': title},
            {'key': 'rank_math_focus_keyword', 'value': title}
        ]
        if SCHEDULE or BACKDATE:
            print(f"Posted Time - {scheduled_date}")
            post.date = scheduled_date
            time_now = scheduled_date
        post.post_status = time_post
        post.id = WP.call(NewPost(post))
        title = title.replace('\n', '')
        print(f"Sukses posted {len(content.split())} Words : {title}")
        if SCHEDULE or BACKDATE:
            scheduled_date = time_now + timedelta(seconds=SLEEP_INTERVAL)
            scheduled_date = scheduled_date.replace(microsecond=0)
    except Exception as e:
        print(e)
        time.sleep(5)

def update_wp():
    max_retries = 3
    retries = 0
    
    while retries < max_retries:        
        if ALL_DATA:
            for item in ALL_DATA:
                update_post(item['title'], item['tag'], item['content'], item['cat'], item['img'])
                print("Posting with login", id)
            break
        else:
            retries += 1
            time.sleep(1)

def is_all_data_valid(all_data):
    if not all_data:
        return False, "ALL_DATA is empty."
    
    for entry in all_data:
        if not isinstance(entry, dict):
            return False, "Entry is not a dictionary."

        required_keys = ['title', 'tag', 'content', 'cat', 'img']
        for key in required_keys:
            if key not in entry:
                return False, f"Missing key: {key}"

            if key == 'img' and not isinstance(entry['img'], str):
                return False, f"Key '{key}' must be a string."

            if key == 'title' and not isinstance(entry['title'], str):
                return False, f"Key '{key}' must be a string."

            if key == 'tag' and not isinstance(entry['tag'], str):
                return False, f"Key '{key}' must be a string."

            if key == 'content' and not isinstance(entry['content'], str):
                return False, f"Key '{key}' must be a string."

            if key == 'cat' and not isinstance(entry['cat'], str):
                return False, f"Key '{key}' must be a string."

    return True, "ALL_DATA is valid."

def check_badwords(keyword):
    list_badwords = [".", "$", "http", "https", "sex", "adult", "dick", "vagina", "blowjob", "fuck", "ass", "shit", "balls", "anal", "hentai", "horny", "porn", "porno", "nude", "gay"]
    keyword_lower = keyword.lower()
    return any(badword in keyword_lower for badword in list_badwords)

if __name__ == '__main__':
    print(Fore.CYAN + f"BOT by SH and GPT4FREE" + Style.RESET_ALL)
    if SCHEDULE or BACKDATE:
        scheduled_date = datetime.strptime(TIME_START, "%Y-%m-%d %H:%M:%S")
    if SPIN_QB:
        tokenqb = qb_login()
        time.sleep(3)
        qb_logout()
        time.sleep(3)
    while True:
        try:
            random.shuffle(login)
            selected_login = random.choice(login)
            id, pw = selected_login.split('|')
            start = time.time()
            ALL_DATA = list()
            while True:
                keyword = process_first_line(keywords)
                keyword = clean_title(keyword)
                is_badword = check_badwords(keyword)
                if is_badword:
                    delete_first_line(keywords)
                else:
                    break
            if keyword:
                print(f"Processing : {keyword}")
                if DEBUG:
                    images = ["https://marketing-assets.surferseo.art/wf-cdn/62666115cfab453aacbd513c/66a78152d262d060fcfaea26_4823dca2-9f51-408f-be12-c8270edce4e3.png@surfer"]
                else:
                    if IMG_GOOGLE:
                        images = google_images(keyword)
                    if IMG_BING:
                        images = bing_images(keyword)
                    if IMG_GOOGLE or IMG_BING:
                        if images is None:
                            print("Failed Scrape images Google or Bing")
                            delete_first_line(keywords)
                            continue                
                        else:
                            print(f"Sukses Scrape {len(images)} Images")
                    else:
                        images = None
                cl = 1
                while True:
                    if cl==3:
                        break
                    else:
                        cl += 1
                    if SPIN_QB:
                        tokenqb = qb_login()
                    if not proxy_resident:
                        get_proxies()
                    content = generate_content(keyword, images)
                    all_content_valid = True 
                    for item in ALL_DATA:
                        wordcounter = item['content']
                        if len(wordcounter.split()) < 500:
                            print("Less Content Detected")
                            all_content_valid = False
                            break
                    if all_content_valid:
                        break
                if cl==3:
                    continue
                is_valid, message = is_all_data_valid(ALL_DATA)
                if is_valid and not DEBUG:
                    update_wp()
                    delete_first_line(keywords)
                else:
                    print()
                    print(message)
                if CSV:
                    csv_file = f"{DOMAIN}.csv"
                    csv_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=ALL_DATA[0].keys())
                        if not csv_exists:
                            writer.writeheader()
                        writer.writerows(ALL_DATA)
            else:
                print("Check Your Keywords.")
                kill_command = ['sudo', 'screen', '-x', 'proxy', '-x', '-X', 'kill']
                subprocess.run(kill_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            end = time.time() - start
            count_lines_in_file()
            print(f'Total Run Time: {time.strftime("%H:%M:%S", time.gmtime(end))}')
            print()
            if SPIN_QB:
                qb_logout()
            if DEBUG:
                break
            else:
                break
                time.sleep(30)
        except Exception as e:
            print(e)
            break
        except KeyboardInterrupt:
            print("\nScript has been terminated. Goodbye!")
            break
