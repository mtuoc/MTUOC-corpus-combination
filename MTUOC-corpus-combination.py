#    MTUOC-NMT-corpus-combination
#    Copyright (C) 2023 Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import importlib
import importlib.util
import codecs
import kenlm
import sqlite3
import gzip
from datetime import datetime
from itertools import (takewhile,repeat)


import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def rawincountgzipped(myfile):
    count=0
    with gzip.open(myfile, 'rb') as f:
        while 1:
            linia=f.readline()
            if not linia:
                break
            count+=1
    return(count)
    
def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

stream = open('config-corpus-combination.yaml', 'r',encoding="utf-8")
config=yaml.load(stream, Loader=yaml.FullLoader)

MTUOC=config["MTUOC"]
sys.path.append(MTUOC)

from MTUOC_train_val_eval import split_corpus


corpusSPE=config["corpusSPE"]
weightSPE=config["weightSPE"]
corpus_GEN=config["corpusGEN"]
weightGEN=config["weightGEN"]
corpusSELECTED=config["corpusSELECTED"]
corpus_GEN_SEL_LINES=int(config["corpus_GEN_SEL_LINES"])
corpus_GEN_MAX_READ=int(config["corpus_GEN_MAX_READ"])

valsize=int(config["valsize"])
evalsize=int(config["evalsize"])

SLcode3=config["SLcode3"]
SLcode2=config["SLcode2"]
TLcode3=config["TLcode3"]
TLcode2=config["TLcode2"]

scores_database_name=config["scores_database_name"]
from_scores_database=config["from_scores_database"]

SL_TOKENIZER=config["SL_TOKENIZER"]
TL_TOKENIZER=config["TL_TOKENIZER"]
tokenize_for_language_model_creation=config["tokenize_for_language_model_creation"]

#VERBOSE
VERBOSE=config["VERBOSE"]
LOGFILE=config["LOG_FILE"]

if VERBOSE:
    logfile=codecs.open(LOGFILE,"w",encoding="utf-8")

if not SL_TOKENIZER=="None":
    if not SL_TOKENIZER.endswith(".py"): SL_TOKENIZER=MTUOC+"/"+SL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', SL_TOKENIZER)
    tokenizerSLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerSLmod)
    tokenizerSL=tokenizerSLmod.Tokenizer()
else:
    tokenizerSL=None

if not TL_TOKENIZER=="None":
    if not TL_TOKENIZER.endswith(".py"): TL_TOKENIZER=MTUOC+"/"+TL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', TL_TOKENIZER)
    tokenizerTLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerTLmod)
    tokenizerTL=tokenizerTLmod.Tokenizer()
else:
    tokenizerTL=None

if VERBOSE:
    cadena="Start of corpus combination: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

if not from_scores_database:
    command="cut -f 1 "+corpusSPE+" > corpusSPESL.temp"
    os.system(command)
    #STEP 1 Tokenize SL SPE corpus (optional, but recommended)
    if VERBOSE:
        cadena="Step 1: Tokenize SL SPE corpus: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")

    if tokenize_for_language_model_creation and not SL_TOKENIZER==None:
        entrada=codecs.open("corpusSPESL.temp","r",encoding="utf-8")
        sortida=codecs.open("corpusSPESLtok.temp","w",encoding="utf-8")
        for linia in entrada:
            linia=linia.rstrip()
            liniatok=tokenizerSL.tokenize(linia)
            sortida.write(liniatok+"\n")
        entrada.close()
        sortida.close()
    else:
        os.rename("corpusSPESL.temp","corpusSPESLtok.temp")
    os.remove("corpusSPESL.temp")

    #STEP 2. Language model creation
    if VERBOSE:
        cadena="Step 2: Language model creation: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    command=MTUOC+"/lmplz -o 5 --skip_symbols --discount_fallback < corpusSPESLtok.temp > lm.arpa."+SLcode2
    os.system(command)
    os.remove("corpusSPESLtok.temp")

    #STEP 3. Binarization of the language model
    if VERBOSE:
        cadena="Step 3: Binarization of the language model: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    command=MTUOC+"/build_binary lm.arpa."+SLcode2+" lm.blm."+SLcode2
    os.system(command)
    os.remove("lm.arpa."+SLcode2)

    #STEP 4. Creation of the SQLite database for perplexities
    if VERBOSE:
        cadena="Step 4: Creation of the SQLite database for perplexities: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    if os.path.isfile(scores_database_name):
        os.remove(scores_database_name)

    conn=sqlite3.connect(scores_database_name)
    cur = conn.cursor() 
    cur.execute("CREATE TABLE perplexities(id INTEGER PRIMARY KEY, perplexity REAL, source TEXT, target TEXT)")
    model=kenlm.Model("lm.blm."+SLcode2) 
    cont=0
    data=[]
    if corpus_GEN.endswith(".gz"):
        cont=0
        with gzip.open(corpus_GEN, 'rb') as f:
            while 1:
                line=f.readline().decode().rstrip()
                if not line:
                    break
                record=[]
                line=line.rstrip()
                camps=line.split("\t")
                source=camps[0]
                source=source.replace("’","'")
                if tokenize_for_language_model_creation:
                    sourcetok=tokenizerSL.tokenize(source)
                else:
                    sourcetok=source
                if len(camps)>=2:
                    target=camps[1]
                else:
                    target=""
                per=model.perplexity(sourcetok)
                record.append(cont)
                record.append(per)
                record.append(source)
                record.append(target)
                data.append(record)
                if cont%1000000==0:
                    cur.executemany("INSERT INTO perplexities (id, perplexity, source, target) VALUES (?,?,?,?)",data)
                    data=[]
                    conn.commit()
                cont+=1
                if not corpus_GEN_MAX_READ==-1 and cont>=corpus_GEN_MAX_READ:
                    break
            cur.executemany("INSERT INTO perplexities (id, perplexity, source, target) VALUES (?,?,?,?)",data)
            conn.commit()               
    else:
        entrada=codecs.open(corpus_GEN,"r",encoding="utf-8")
        cont=0
        for line in entrada:
            record=[]
            line=line.rstrip()
            camps=line.split("\t")
            source=camps[0]
            source=source.replace("’","'")
            if tokenize_for_language_model_creation:
                sourcetok=tokenizerSL.tokenize(source)
            else:
                sourcetok=source
            if len(camps)>=2:
                target=camps[1]
            else:
                target=""
            per=model.perplexity(sourcetok)
            record.append(cont)
            record.append(per)
            record.append(source)
            record.append(target)
            data.append(record)
            if cont%1000000==0:
                cur.executemany("INSERT INTO perplexities (id, perplexity, source, target) VALUES (?,?,?,?)",data)
                data=[]
                conn.commit()
            cont+=1
            if not corpus_GEN_MAX_READ==-1 and cont>=corpus_GEN_MAX_READ:
                break
        cur.executemany("INSERT INTO perplexities (id, perplexity, source, target) VALUES (?,?,?,?)",data)
        conn.commit()

else:
    conn=sqlite3.connect(scores_database_name)
    cur = conn.cursor()

if VERBOSE:
    cadena="Step 5: Creation of the selected corpus from the general corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

cur.execute("SELECT id,perplexity,source,target FROM perplexities ORDER BY perplexity ASC limit "+str(corpus_GEN_SEL_LINES)+";")
results=cur.fetchall()

sortida=codecs.open(corpusSELECTED,"w",encoding="utf-8")
for result in results:
    cadena=result[2]+"\t"+result[3]
    sortida.write(cadena+"\n")
sortida.close()

entrada=codecs.open(corpusSPE,"r",encoding="utf-8")
sortida=codecs.open("corpusSPEPAR.tmp","w",encoding="utf-8")

#Ensuring parallel segments
contparallel=0
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        cadena=camps[0]+"\t"+camps[1]+"\t"+str(weightSPE)
        sortida.write(cadena+"\n")
        contparallel+=1

entrada.close()
sortida.close()

if VERBOSE:
    cadena="Step 6: Splitting corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
split_corpus("corpusSPEPAR.tmp",valsize,evalsize,SLcode3,TLcode3)

trainCorpus="train-"+SLcode3+"-"+TLcode3+".txt"
valCorpus="val-"+SLcode3+"-"+TLcode3+".txt"
evalCorpus="eval-"+SLcode3+"-"+TLcode3+".txt"
trainPreCorpus="train-pre-"+SLcode3+"-"+TLcode3+".txt"
valPreCorpus="val-pre-"+SLcode3+"-"+TLcode3+".txt"



lenval=rawincount(valCorpus)
leneval=rawincount(evalCorpus)


entrada=codecs.open(corpusSELECTED,"r",encoding="utf-8")
sortidaTrain=codecs.open(trainCorpus,"a",encoding="utf-8")
sortidaVal=codecs.open(valCorpus,"a",encoding="utf-8")
sortidaEval=codecs.open(evalCorpus,"a",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    if lenval<valsize:
        sortidaVal.write(linia+"\t"+str(weightGEN)+"\n")
        lenval+=1
    elif leneval<evalsize:
        sortidaEval.write(linia+"\t"+str(weightGEN)+"\n")
        leneval+=1
    else:
        sortidaTrain.write(linia+"\t"+str(weightGEN)+"\n")

sortidaTrain.close()
sortidaVal.close()
sortidaEval.close()

entrada=codecs.open(evalCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open("eval."+SLcode2,"w",encoding="utf-8")
sortidaTL=codecs.open("eval."+TLcode2,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
        
entrada.close()
sortidaSL.close()        
sortidaTL.close()
