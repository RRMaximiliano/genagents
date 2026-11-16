import json
import os
import subprocess
import tempfile
import threading
import time
import base64
from typing import List, Union

from simulation_engine import settings as _settings
from simulation_engine.settings import *

# Backend settings. Defaults keep the original OpenAI behavior for older
# settings.py files that don't define them.
LLM_BACKEND = getattr(_settings, "LLM_BACKEND", "openai")
CLAUDE_MODEL = getattr(_settings, "CLAUDE_MODEL", None)
CODEX_MODEL = getattr(_settings, "CODEX_MODEL", None)
CLI_TIMEOUT = getattr(_settings, "CLI_TIMEOUT", 300)
EMBEDDING_BACKEND = getattr(_settings, "EMBEDDING_BACKEND", "openai")
LOCAL_EMBEDDING_MODEL = getattr(_settings, "LOCAL_EMBEDDING_MODEL", 
                                "sentence-transformers/all-MiniLM-L6-v2")

if LLM_BACKEND not in ("openai", "claude_cli", "codex_cli"):
  raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")
if EMBEDDING_BACKEND not in ("openai", "local"):
  raise ValueError(f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}")

if LLM_BACKEND == "openai" or EMBEDDING_BACKEND == "openai":
  import openai
  openai.api_key = OPENAI_API_KEY


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ##################### [SECTION 2: SUBSCRIPTION CLIs] #######################
# ============================================================================

# If these are set, the CLIs may bill the API instead of the subscription.
_CLI_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", 
                     "OPENAI_API_KEY", "CODEX_API_KEY")

# Headless, no tools, no MCP servers, no user/project settings or hooks.
_CLAUDE_FLAGS = ["-p", "--tools", "", "--strict-mcp-config", 
                 "--setting-sources", "", "--disable-slash-commands", 
                 "--no-session-persistence", 
                 "--system-prompt", "Answer the user's request directly."]

# Non-interactive, read-only sandbox, nothing persisted, tools disabled.
_CODEX_FLAGS = ["exec", "--sandbox", "read-only", "--ephemeral", 
                "--skip-git-repo-check", "--ignore-user-config", 
                "--ignore-rules", "--color", "never", 
                "-c", 'web_search="disabled"']
for _feature in ("shell_tool", "unified_exec", "code_mode_host", "apps", 
                 "plugins", "multi_agent", "image_generation", "browser_use", 
                 "computer_use", "in_app_browser", "hooks"): 
  _CODEX_FLAGS += ["--disable", _feature]


def _run_cli(cmd: List[str], stdin_text: str, cwd: str) -> str:
  """Run a CLI backend with API keys stripped from its environment."""
  env = {k: v for k, v in os.environ.items() if k not in _CLI_STRIPPED_ENV}
  try:
    result = subprocess.run(cmd, input=stdin_text, capture_output=True,
                            text=True, cwd=cwd, env=env, timeout=CLI_TIMEOUT)
  except subprocess.TimeoutExpired:
    raise RuntimeError(f"{cmd[0]} timed out after {CLI_TIMEOUT}s")
  if result.returncode != 0:
    detail = (result.stderr.strip() or result.stdout.strip())[-1000:]
    raise RuntimeError(f"{cmd[0]} exited with {result.returncode}: {detail}")
  return result.stdout


def _image_media_type(raw: bytes) -> str:
  if raw.startswith(b"\x89PNG"): return "image/png"
  if raw.startswith(b"GIF8"): return "image/gif"
  if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP": return "image/webp"
  return "image/jpeg"


def _claude_request(prompt: str, images: List[bytes] = ()) -> str:
  cmd = ["claude", *_CLAUDE_FLAGS]
  if CLAUDE_MODEL:
    cmd += ["--model", CLAUDE_MODEL]

  with tempfile.TemporaryDirectory() as tmp:
    if not images:
      return _run_cli(cmd + ["--output-format", "text"], prompt, tmp).strip()

    # Images go inline as base64 content blocks, so no file tools are needed.
    content = [{"type": "text", "text": prompt}]
    for raw in images:
      content.append({"type": "image", "source": {
        "type": "base64", "media_type": _image_media_type(raw), 
        "data": base64.b64encode(raw).decode("utf-8")}})
    message = json.dumps({"type": "user", 
                          "message": {"role": "user", "content": content}})
    cmd += ["--input-format", "stream-json", 
            "--output-format", "stream-json", "--verbose"]
    output = _run_cli(cmd, message, tmp)

  for line in reversed(output.strip().splitlines()):
    event = json.loads(line)
    if event.get("type") == "result":
      if event.get("is_error"):
        raise RuntimeError(f"claude: {event.get('result')}")
      return event["result"].strip()
  raise RuntimeError("claude returned no result")


def _codex_request(prompt: str, images: List[bytes] = ()) -> str:
  with tempfile.TemporaryDirectory() as tmp:
    out_file = os.path.join(tmp, "last_message.txt")
    cmd = ["codex", *_CODEX_FLAGS, "-C", tmp, "-o", out_file]
    if CODEX_MODEL:
      cmd += ["--model", CODEX_MODEL]
    cmd.append("-")  # read the prompt from stdin
    for count, raw in enumerate(images):
      image_file = os.path.join(
        tmp, f"image_{count}.{_image_media_type(raw).split('/')[1]}")
      with open(image_file, "wb") as f:
        f.write(raw)
      cmd += ["--image", image_file]

    _run_cli(cmd, prompt, tmp)
    with open(out_file) as f:
      return f.read().strip()


def _cli_request(prompt: str, images: List[bytes] = ()) -> str:
  if LLM_BACKEND == "claude_cli":
    return _claude_request(prompt, images)
  return _codex_request(prompt, images)


# ============================================================================
# ####################### [SECTION 3: SAFE GENERATE] #########################
# ============================================================================

def gpt_request(prompt: str, 
                model: str = "gpt-4o", 
                max_tokens: int = 1500) -> str:
  """Make a request to the configured LLM backend. With a CLI backend, 
     `model` and `max_tokens` are ignored (see CLAUDE_MODEL / CODEX_MODEL)."""
  if LLM_BACKEND != "openai":
    try:
      return _cli_request(prompt)
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"

  if model == "o1-preview": 
    try:
      client = openai.OpenAI(api_key=OPENAI_API_KEY)
      response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
      )
      return response.choices[0].message.content
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"

  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def gpt4_vision(messages: List[dict], max_tokens: int = 1500) -> str:
  """Make a request to OpenAI's GPT-4 Vision model (or a CLI backend)."""
  if LLM_BACKEND != "openai":
    try:
      texts, images = [], []
      for message in messages:
        content = message["content"]
        if isinstance(content, str):
          content = [{"type": "text", "text": content}]
        for part in content:
          if part["type"] == "text":
            texts.append(part["text"])
          elif part["type"] == "image_url":
            data = part["image_url"]["url"].split("base64,", 1)[1]
            images.append(base64.b64decode(data))
      return _cli_request("\n\n".join(texts), images)
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"

  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  if file_attachment and file_type:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    messages = [{"role": "user", "content": prompt}]

    if file_type.lower() == 'image':
      with open(file_attachment, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
      messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please refer to the attached image."},
            {"type": "image_url", "image_url": 
              {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
      })
      response = gpt4_vision(messages, max_tokens)

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      response = gpt_request(prompt, gpt_version, max_tokens)

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    for i in range(repeat):
      response = gpt_request(prompt, model=gpt_version)
      if response != "GENERATION ERROR":
        break
      time.sleep(2**i)
    else:
      response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 4: OTHER API FUNCTIONS] ######################
# ============================================================================

_local_embedder = None
_local_embedder_lock = threading.Lock()


def _get_local_embedder():
  """Lazy-load the sentence-transformers model (shared across threads)."""
  global _local_embedder
  with _local_embedder_lock:
    if _local_embedder is None:
      from sentence_transformers import SentenceTransformer
      _local_embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
  return _local_embedder


def get_text_embedding(text: str, 
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using OpenAI's API, or a local
     sentence-transformers model when EMBEDDING_BACKEND = "local"."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  text = text.replace("\n", " ").strip()
  if EMBEDDING_BACKEND == "local":
    return _get_local_embedder().encode(text).tolist()

  response = openai.embeddings.create(
    input=[text], model=model).data[0].embedding
  return response









