# -*- coding: utf-8 -*-
import os
import glob
import codecs
import spacy
import utils_nlp
import json
from pycorenlp import StanfordCoreNLP
import sys


def get_sentences_and_tokens_from_pos_tagger(pos_tags):
    sentences = []
    sentence = []
    for pos_tag in pos_tags:
        if ' ' not in pos_tag['text']:
            token_dict = {}
            token_dict['start'] = pos_tag['start']
            token_dict['end'] = pos_tag['end'] 
            token_dict['text'] = pos_tag['text']
            if token_dict['text'] in ['\n', '\t', ' ', '']:
                continue
            sentence.append(token_dict)
            if pos_tag['text'] == '.' or  pos_tag['text'] == '!' or pos_tag['text'] == '?':
                sentences.append(sentence)
                sentence = []
        else:
            i = pos_tag['start']
            current_token = ''
            current_start = pos_tag['start']
            while i < pos_tag['end']:
                try:
                    current_char = pos_tag['text'][i-pos_tag['start']]
                except:
                    token_dict = {}
                    token_dict['start'] = current_start
                    token_dict['end'] = i+1
                    token_dict['text'] = pos_tag['text'][current_start-pos_tag['start']:i+1-pos_tag['start']]
                    break
                if (current_char == ' ' and current_token != '') or i+1 == pos_tag['end']:
                    token_dict = {}
                    token_dict['start'] = current_start
                    token_dict['end'] = i+1
                    token_dict['text'] = pos_tag['text'][current_start-pos_tag['start']:i+1-pos_tag['start']]
                    sentence.append(token_dict)
                    current_token = ''
                    if token_dict['text'] == '.' or  token_dict['text'] == '!' or token_dict['text'] == '?':
                        sentences.append(sentence)
                        sentence = []
                elif current_char != ' ':
                    if current_token == '':
                        current_start = i
                    current_token += current_char

                i += 1
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'], 
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    if type(output) is str:
        output = json.loads(output, strict=False)
    return output

def get_sentences_and_tokens_from_stanford(text, core_nlp):
    stanford_output = get_stanford_annotations(text, core_nlp)
    sentences = []
    for sentence in stanford_output['sentences']:
        tokens = []
        for token in sentence['tokens']:
            token['start'] = int(token['characterOffsetBegin'])
            token['end'] = int(token['characterOffsetEnd'])
            token['text'] = text[token['start']:token['end']]
            if token['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token['text'].split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token['text'], 
                                                                                                                           token['text'].replace(' ', '-')))
                token['text'] = token['text'].replace(' ', '-')
            tokens.append(token)
        sentences.append(tokens)
    return sentences

def get_entities_from_brat(text_filepath, annotation_filepath, verbose=False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text =f.read()
    if verbose: print("\ntext:\n{0}\n".format(text))
    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = {}
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                if verbose:
                    print("entity: {0}".format(entity))
                # Check compatibility between brat text and annotation
                if utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                    utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                # add to entitys data
                entities.append(entity)
    if verbose: print("\n\n")
    return text, entities

def get_pos_tags_from_brat(text,annotation_filepath2, verbose=False):
    # parse annotation file
    pos_tags = []
    with codecs.open(annotation_filepath2, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                pos_tag = {}
                pos_tag['id'] = id_anno
                pos_tag['type'] = anno[1] # tag
                pos_tag['start'] = int(anno[2])
                pos_tag['end'] = int(anno[3])
                pos_tag['text'] = ' '.join(anno[4:])
                if verbose:
                    print("pos_tag: {0}".format(pos_tag))
                # add to entity data
                pos_tags.append(pos_tag)
    if verbose: print("\n\n")
    
    return pos_tags

def check_brat_annotation_and_text_compatibility(brat_folder):
    '''
    Check if brat annotation and text files are compatible.
    '''
    dataset_type =  os.path.basename(brat_folder)
    print("Checking the validity of BRAT-formatted {0} set... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(brat_folder, '*.txt')))
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # check if annotation file exists
        if not os.path.exists(annotation_filepath):
            raise IOError("Annotation file does not exist: {0}".format(annotation_filepath))
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
    print("Done.")

def brat_to_conll(input_folder, output_filepath, tokenizer, language):
    '''
    Assumes '.txt' and '.ann' files are in the input_folder.
    Checks for the compatibility between .txt and .ann at the same time.
    '''
    use_pos = False
    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language)
    elif tokenizer == 'stanford':
        core_nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    elif tokenizer == 'pos':
        use_pos = True
    else:
        raise ValueError("tokenizer should be either 'spacy' or 'stanford'.")
    verbose = False
    dataset_type =  os.path.basename(input_folder)
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')

        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()

        if use_pos:
            annotation_filepath2 = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann2')
            # create annotation file if it does not exist
            if not os.path.exists(annotation_filepath2):
                codecs.open(annotation_filepath2, 'w', 'UTF-8').close()

        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity:entity["start"])

        if use_pos:
            pos_tags = get_pos_tags_from_brat(text,annotation_filepath2)
            sentences = get_sentences_and_tokens_from_pos_tagger(pos_tags)
        else:
            if tokenizer == 'spacy':
                sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)
            elif tokenizer == 'stanford':
                sentences = get_sentences_and_tokens_from_stanford(text, core_nlp)
        
        if use_pos:
            token_counter = 0
            rep_pos_max = 0
            rep_pos_counter = 0
            rep_pos = False
        for sentence in sentences:
            inside = False
            previous_token_label = 'O'
            for token in sentence:
                token['label'] = 'O'
                for entity in entities:
                    if entity['start'] <= token['start'] < entity['end'] or \
                       entity['start'] < token['end'] <= entity['end'] or \
                       token['start'] < entity['start'] < entity['end'] < token['end']:

                        token['label'] = entity['type'].replace('-', '_') # Because the ANN doesn't support tag with '-' in it

                        break
                    elif token['end'] < entity['start']:
                        break
                        
                if len(entities) == 0:
                    entity={'end':0}
                if token['label'] == 'O':
                    gold_label = 'O'
                    inside = False
                elif inside and token['label'] == previous_token_label:
                    gold_label = 'I-{0}'.format(token['label'])
                else:
                    inside = True
                    gold_label = 'B-{0}'.format(token['label'])
                if token['end'] == entity['end']:
                    inside = False
                previous_token_label = token['label']
                if use_pos:
                    pos_tag = pos_tags[token_counter]['type']
                    if not rep_pos and len(pos_tags[token_counter]['text'].split()) > 1:
                        rep_pos = True
                        rep_pos_max = len(pos_tags[token_counter]['text'].split())
                        rep_pos_counter = 0
                    elif rep_pos:
                        rep_pos_counter += 1
                        if rep_pos_counter >= rep_pos_max:
                            rep_pos = False
                            rep_pos_counter = 0
                    else:
                        token_counter += 1
                    if len('{0} {1} {2} {3} {4} {5}\n'.format(token['text'], base_filename, token['start'], token['end'],pos_tag,gold_label).split()) != 6:
                        continue
                        input('{0} {1} {2} {3} {4} {5}\n'.format(token['text'], base_filename, token['start'], token['end'],pos_tag,gold_label))

                    if verbose: print('{0} {1} {2} {3} {4} {5}\n'.format(token['text'].split()[0], base_filename, token['start'], token['end'],pos_tag,gold_label))
                    output_file.write('{0} {1} {2} {3} {4} {5}\n'.format(token['text'].split()[0], base_filename, token['start'], token['end']-(len(token['text']) - len(token['text'].split()[0])),pos_tag,gold_label))
                else:
                    if verbose: print('{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'], gold_label))
                    output_file.write('{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'], gold_label))
            if verbose: print('\n')
            output_file.write('\n')

    output_file.close()
    print('Done.')
    if not use_pos:
        if tokenizer == 'spacy':
            del spacy_nlp
        elif tokenizer == 'stanford':
            del core_nlp
