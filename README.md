# llama7b_cpu
My attempt at running quantized 7b llama model on CPU

I have just followed directions in the Readme file and was able to run quantized llama 7b model. However my token generation was very slow. I do have an old Dell XPS, that runs linux debian.

Step 1: Looks like the repo maintainer has spent a good amount of time trying to develelop quantization of meta models which prettey much allows it to run on most computers

    git clone https://github.com/ggerganov/llama.cpp

Step 2: Obtain the original models. In this case 7B and place it in 'models' folder
How to get it you ask? I used deluge torrent on linux with hash of cdee3052d85c697b84f4c1192f43a2276c0daea0
There are of course other ways on Hugging face.

Once you get the models, place the 7B model in the models folder of llama.cpp. Also please copy the tokenizer.model and tokenizer_checklist.chk in the llama.cpp folder.

    cd llama.cpp

Step 3: In my case I had python3 and used it to install dependencies

    python3 -m pip install -r requirements.txt

Step 4: Now to convert to ggml FP16 format
Why this format you ask? GGML FP16 format refers to a specific format for representing numerical data in machine learning models. FP16 stands for 16-bit floating-point, which is a data type commonly used in deep learning frameworks for training and inference.

In machine learning, numerical values are typically stored as floating-point numbers, which can represent both fractional and whole numbers with a certain level of precision. The precision of a floating-point number is determined by the number of bits allocated for the fractional and exponent parts.

FP16 format uses 16 bits to represent a floating-point number, with 1 bit for the sign, 5 bits for the exponent, and 10 bits for the significand (also known as mantissa). Compared to the more commonly used FP32 format (32-bit floating-point), FP16 has a reduced range and precision, but it requires half the memory.

The GGML (Generalized GPU Machine Learning) framework is an AI accelerator library developed by NVIDIA. It is designed to optimize and accelerate machine learning workloads on NVIDIA GPUs. GGML supports various numerical formats, including FP16, to enable faster computations and reduce memory usage in deep learning models.

Using the FP16 format in machine learning models can provide benefits such as reduced memory footprint and faster training and inference speeds. However, it is important to note that using lower precision formats like FP16 can lead to a loss of numerical precision, which may impact the accuracy and stability of the model. 

    python3 convert.py models/7B/

Step 4: Now to quantize the model to 4 bits. This allows it to run on a simple CPU like mine:

    ./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0

Step 6: Now to start the chat mode you might have seen online.

    ./examples/chat.sh

If you just want to run the interface:

    ./main -m ./models/7B/ggml-model-q4_0.bin -n 128


**Other things to know:**

**Memory requirements for the models:**

As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| Model | Original size | Quantized size (4-bit) |
|------:|--------------:|-----------------------:|
|    7B |         13 GB |                 3.9 GB |
|   13B |         24 GB |                 7.8 GB |


**Starting up:**
When starting up it looks like this:

    main: build = 803 (1d656d6)
    main: seed  = 1691487030
    llama.cpp: loading model from ./models/7B/ggml-model-q4_0.bin
    llama_model_load_internal: format     = ggjt v3 (latest)
    llama_model_load_internal: n_vocab    = 32000
    llama_model_load_internal: n_ctx      = 512
    llama_model_load_internal: n_embd     = 4096
    llama_model_load_internal: n_mult     = 256
    llama_model_load_internal: n_head     = 32
    llama_model_load_internal: n_layer    = 32
    llama_model_load_internal: n_rot      = 128
    llama_model_load_internal: ftype      = 2 (mostly Q4_0)
    llama_model_load_internal: n_ff       = 11008
    llama_model_load_internal: model size = 7B
    llama_model_load_internal: ggml ctx size =    0.08 MB
    llama_model_load_internal: mem required  = 5439.94 MB (+ 1026.00 MB per state)
    llama_new_context_with_model: kv self size  =  256.00 MB

    system_info: n_threads = 4 / 4 | AVX = 1 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 
    main: interactive mode on.
    Reverse prompt: 'User:'
    sampling: repeat_last_n = 64, repeat_penalty = 1.000000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
    generate: n_ctx = 512, n_batch = 512, n_predict = 256, n_keep = 48


    == Running in interactive mode. ==
     - Press Ctrl+C to interject at any time.
     - Press Return to return control to LLaMa.
     - To return control without starting a new line, end your input with '/'.
     - If you want to submit another line, end your input with '\'.

     Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

    User: Hello, Bob.
    Bob: Hello. How may I help you today?
    User: Please tell me the largest city in Europe.
    Bob: Sure. The largest city in Europe is Moscow, the capital of Russia._

**Do I have the right models?** Run checksums

    ./scripts/verify-checksum-models.py
