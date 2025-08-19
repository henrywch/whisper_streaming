#!/usr/bin/env python3
from sympy.physics.units import current

from whisper_online import *

import sys
import argparse
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43001)
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others. 
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)


######### Server objects

import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


import io
import soundfile

from datetime import datetime

import threading

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

        self.new_session = True
        self.outf_name = str()

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            format_line = "%1.0f %1.0f %s" % (beg,end,o[2])
            print(format_line ,flush=True,file=sys.stderr)

            if self.new_session:
                self.outf_name= "results\LIVE_{}.txt".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                self.new_session = False
            with open(self.outf_name, mode="a") as f:
                f.write("%s " % o[2])
                f.flush()

            return format_line
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)


######### The Talking Codes

import csv
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

from kokoro import KPipeline
import sounddevice as sd

class MockingBird:

    def __init__(self, model_path):
        self.model_path = model_path
        self.load_llm()
        self.model = None
        self.tokenizer = None

        self.ans_folder = "results\ANS_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.ans_file_name = os.path.join(self.ans_folder, "chat_history.tsv")

    def load_llm(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def locate_dialogs(self, local_dir: str = "results"):
        search_pattern = os.path.join(local_dir, 'LIVE_*.txt')
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            return None
        last_file_path = sorted(matching_files)[-1]

        return os.path.basename(last_file_path)

    def load_chat_history(self):
        history = []
        if os.path.exists(self.ans_file_name) and os.stat(self.ans_file_name).st_size > 0:
            with open(self.ans_file_name, "r", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    history.append({"role": "user", "content": row['user']})
                    history.append({"role": "assistant", "content": row['bot']})
        return history

    def llm_thinking(self):
        dialog_file_name = self.locate_dialogs()

        if not dialog_file_name:
            print("No new dialog file found. Exiting.")
            return

        with open(dialog_file_name, "r") as f:
            dialogs = f.read().strip()

        messages = self.load_chat_history()

        messages.insert(0, {"role": "system", "content": """
                                          You are Qwen, created by Alibaba Cloud. You are a helpful assistant. \
                                          Now someone likes to talk you, he/she can be anybody. Just calm down \
                                          and talk to him/her according to what you are trained. Don't let them \
                                          find out that YOU are a ROBOT.
                                          The consequence can be DISASTROUS!!!"""
                            })
        messages.append({"role": "user", "content": dialogs})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        file_exists = os.path.exists(self.ans_file_name)

        with open(self.ans_file_name, "a", newline='', encoding='utf-8') as f:
            fieldnames = ['user', 'bot']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')

            if not file_exists or os.stat(self.ans_file_name).st_size == 0:
                writer.writeheader()

            writer.writerow({'user': dialogs, 'bot': response})

        self.kokoro_output(response)

    def kokoro_output(self, response: str = "Sorry, pardon me?"):
        if not os.path.exists(self.ans_folder):
            os.makedirs(self.ans_folder)

        pipeline = KPipeline(lang_code='b')
        generator = pipeline(response, voice='af_heart')

        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Generating part {i}: {gs}, {ps}")
            sd.play(audio, samplerate=24000)
            sd.wait()

            sf.write(os.path.join(self.ans_folder, f'{i}.wav'), audio, 24000)

# server loop
def server_loop(se):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info('Listening on'+str((args.host, args.port)))
        while not se.is_set():
            conn, addr = s.accept()
            logger.info('Connected to client on {}'.format(addr))
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, args.min_chunk_size)
            proc.process()
            conn.close()
            logger.info('Connection to client closed')

    logger.info('Connection closed, terminating.')


if __name__ == '__main__':
    mockbd = MockingBird(".\models\Qwen")

    stop_event = threading.Event()
    server_thread = threading.Thread(target=server_loop, args=(stop_event,))
    server_thread.daemon = True
    server_thread.start()

    try:
        while True:
            key = input()
            if key == 'Q':
                stop_event.set()
                print("Stopping server...")
                break
    except KeyboardInterrupt:
        stop_event.set()
        print("\nServer stopped by KeyboardInterrupt.")

    server_thread.join(timeout=2.0)  # Wait for server thread to exit
    sys.exit(0)
