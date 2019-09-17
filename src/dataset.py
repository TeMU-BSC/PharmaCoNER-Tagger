import sklearn.preprocessing
import utils
import collections
import codecs
import utils_nlp
import re
import time
import token
import os
import pickle
import random


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def _parse_gaz(self,gaz_filepath):
        with open(gaz_filepath) as f:
            s = f.read()
        lines = s.splitlines()
        gaz_set = set([])
        for l in lines:
            gaz_set.add(l)
        self.gaz_set = gaz_set

    def _parse_aff(self, aff_filepath):

        with open(aff_filepath) as f:
            s = f.read()
        lines = s.splitlines()
        aff_set = dict(suffix=[], prefix=[], root=[])
        for l in lines:
            tmp = l.split('\n')
            if tmp[0].strip() == 'suffix':
                aff_set['suffix'].append(tmp[2].strip())
            elif tmp[0].strip() == 'prefix':
                aff_set['prefix'].append(tmp[2].strip())
            elif tmp[0].strip() == 'root':
                aff_set['root'].append(tmp[2].strip())

        self.aff_set = aff_set

    def _parse_dataset(self, dataset_filepath, parameters):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)
        if parameters['use_pos']:
            pos_tag_count = collections.defaultdict(lambda: 0)
        if parameters['use_gaz']:
            gaz_count = collections.defaultdict(lambda: 0)
            #self._parse_gaz(parameters['gaz_filepath'])
        if parameters['use_aff']:
            aff_count = collections.defaultdict(lambda: 0)

        line_count = -1
        tokens = []
        labels = []
        pos_tags = []
        new_token_sequence = []
        new_label_sequence = []
        if parameters['use_pos']:
            new_pos_tag_sequence = []
        if parameters['use_gaz']:
            new_gaz_sequence = []
            gazs = []
        if parameters['use_aff']:
            new_aff_sequence = []
            affs = []
        if dataset_filepath:
            f = codecs.open(dataset_filepath, 'r', 'UTF-8')
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels.append(new_label_sequence)
                        tokens.append(new_token_sequence)
                        if parameters['use_pos']:
                            pos_tags.append(new_pos_tag_sequence)
                        if parameters['use_gaz']:
                            gazs.append(new_gaz_sequence)
                        if parameters['use_aff']:
                            affs.append(new_aff_sequence)
                            new_aff_sequence = []
                        new_token_sequence = []
                        new_label_sequence = []
                        new_pos_tag_sequence = []
                        new_gaz_sequence = []
                    continue
                token = str(line[0])
                label = str(line[-1])
                # beware: in both cases we are assuming bioes
                if parameters['use_pos']:
                    '''
                    if parameters['tokenizer'] == 'pos':
                        pos_tag = str(line[-2])
                    else:
                        pos_tag = str(line[-3])
                    '''
                    if parameters['tokenizer'] == 'pos':
                        pos_tag = str(line[-3])
                    else:
                        pos_tag = str(line[-4])
                    #print(pos_tag)
                if parameters['use_gaz']:
                    gaz = token.lower() in self.gaz_set
                    if gaz:
                        gaz = 1
                    else:
                        gaz = 0
                if parameters['use_aff']:
                    aff = 0
                    # Check for prefix
                    for pref in self.aff_set['prefix']:
                        pattern = '^' + re.escape(pref.lower())
                        result = re.match(pattern, token.lower())
                        if result:
                            aff = 1

                    for suf in self.aff_set['suffix']:
                        pattern = re.escape(suf.lower()) + '$'
                        result = re.match(pattern, token.lower())
                        if result:
                            aff = 1

                    for rot in self.aff_set['root']:
                        result = token.lower().find(rot)
                        if result > 1:
                            aff = 1


                token_count[token] += 1
                label_count[label] += 1
                if parameters['use_pos']:
                    pos_tag_count[pos_tag] += 1

                if parameters['use_gaz']:
                    gaz_count[gaz] += 1

                if parameters['use_aff']:
                    aff_count[aff] += 1

                new_token_sequence.append(token)
                new_label_sequence.append(label)
                if parameters['use_pos']:
                    new_pos_tag_sequence.append(pos_tag)
                if parameters['use_gaz']:
                    new_gaz_sequence.append(gaz)
                if parameters['use_aff']:
                    new_aff_sequence.append(aff)

                for character in token:
                    character_count[character] += 1

                if self.debug and line_count > 200: break# for debugging purposes

            if len(new_token_sequence) > 0:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
                if parameters['use_pos']:
                    pos_tags.append(new_pos_tag_sequence)
                if parameters['use_gaz']:
                    gazs.append(new_gaz_sequence)
                if parameters['use_aff']:
                    affs.append(new_aff_sequence)
            f.close()
        if not parameters['use_pos']:
            pos_tags = None
            pos_tag_count = None
        if not parameters['use_gaz']:
            gazs = None
            gaz_count = None
        if not parameters['use_aff']:
            affs = None
            aff_count = None
        return labels, tokens, token_count, label_count, character_count, pos_tags, pos_tag_count, gazs, gaz_count, affs, aff_count


    def _convert_to_indices(self, dataset_types, parameters):
        tokens = self.tokens
        labels = self.labels
        if parameters['use_pos']:
            pos_tags = self.pos_tags
        if parameters['use_gaz']:
            gazs = self.gazs
        if parameters['use_aff']:
            affs = self.affs
        token_to_index = self.token_to_index
        character_to_index = self.character_to_index
        label_to_index = self.label_to_index
        index_to_label = self.index_to_label
        if parameters['use_pos']:
            index_to_pos_tag = self.index_to_pos_tag
            pos_tag_to_index = self.pos_tag_to_index
        if parameters['use_gaz']:
            gaz_to_index = self.gaz_to_index
        if parameters['use_aff']:
            aff_to_index = self.aff_to_index
        
        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        characters = {}
        if parameters['use_pos']:
            pos_tag_indices = {}
        if parameters['use_gaz']:
            gaz_indices = {}
        if parameters['use_aff']:
            aff_indices = {}
        token_lengths = {}
        character_indices = {}
        character_indices_padded = {}
        for dataset_type in dataset_types:
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            #if parameters['use_pos']:
            #    pos_tags[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            character_indices_padded[dataset_type] = []
            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([token_to_index.get(token, self.UNK_TOKEN_INDEX) for token in token_sequence])
                characters[dataset_type].append([list(token) for token in token_sequence])
                character_indices[dataset_type].append([[character_to_index.get(character, random.randint(1, max(self.index_to_character.keys()))) for character in token] for token in token_sequence])
                token_lengths[dataset_type].append([len(token) for token in token_sequence])
                longest_token_length_in_sequence = max(token_lengths[dataset_type][-1])
                character_indices_padded[dataset_type].append([utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX) for temp_token_indices in character_indices[dataset_type][-1]])

            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])

            if parameters['use_pos']:
                pos_tag_indices[dataset_type] = []
                for pos_tag_sequence in pos_tags[dataset_type]:
                    pos_tag_indices[dataset_type].append([pos_tag_to_index[pos_tag] for pos_tag in pos_tag_sequence])

            if parameters['use_gaz']:
                gaz_indices[dataset_type] = []
                for gaz_sequence in gazs[dataset_type]:
                    gaz_indices[dataset_type].append([gaz_to_index[gaz] for gaz in gaz_sequence])

            if parameters['use_aff']:
                aff_indices[dataset_type] = []
                for aff_sequence in affs[dataset_type]:
                    aff_indices[dataset_type].append([aff_to_index[aff] for aff in aff_sequence])
        
        if self.verbose:
            print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths['train'][0][0:10]))
        if self.verbose:
            print('characters[\'train\'][0][0:10]: {0}'.format(characters['train'][0][0:10]))
        if self.verbose:
            print('token_indices[\'train\'][0:10]: {0}'.format(token_indices['train'][0:10]))
        if self.verbose:
            print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))
        if self.verbose:
            print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices['train'][0][0:10]))
        if self.verbose:
            print('character_indices_padded[\'train\'][0][0:10]: {0}'.format(character_indices_padded['train'][0][0:10])) # Vectorize the labels
        if parameters['use_pos']:
            if self.verbose:
                print('pos_tag_indices[\'train\'][0:10]: {0}'.format(pos_tag_indices['train'][0:10]))
        if parameters['use_gaz']:
            if self.verbose:
                print('gaz_indices[\'train\'][0:10]: {0}'.format(gaz_indices['train'][0:10]))
        if parameters['use_aff']:
            if self.verbose:
                print('aff_indices[\'train\'][0:10]: {0}'.format(aff_indices['train'][0:10]))
        # [Numpy 1-hot array](http://stackoverflow.com/a/42263603/395857)
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(index_to_label.keys()) + 1))
        label_vector_indices = {}
        for dataset_type in dataset_types:
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(label_binarizer.transform(label_indices_sequence))

        if parameters['use_pos']:
            pos_tag_binarizer = sklearn.preprocessing.LabelBinarizer()
            pos_tag_binarizer.fit(range(max(index_to_pos_tag.keys()) + 1))
            pos_tag_vector_indices = {}
            for dataset_type in dataset_types:
                pos_tag_vector_indices[dataset_type] = []
                for pos_tag_indices_sequence in pos_tag_indices[dataset_type]:
                    pos_tag_vector_indices[dataset_type].append(pos_tag_binarizer.transform(pos_tag_indices_sequence))
        if parameters['use_gaz']:
            gaz_vector_indices = {}
            for dataset_type in dataset_types:
                gaz_vector_indices[dataset_type] = []
                for gaz_indices_sequence in gaz_indices[dataset_type]:
                    gaz_vector_index = []
                    for element in gaz_indices_sequence:
                        gaz_vector_index.append([element])
                    gaz_vector_indices[dataset_type].append(gaz_vector_index)
        if parameters['use_aff']:
            aff_vector_indices = {}
            for dataset_type in dataset_types:
                aff_vector_indices[dataset_type] = []
                for aff_indices_sequence in aff_indices[dataset_type]:
                    aff_vector_index = []
                    for element in aff_indices_sequence:
                        aff_vector_index.append([element])
                    aff_vector_indices[dataset_type].append(aff_vector_index)
        if self.verbose:
            print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))
        if self.verbose:
            print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))
        
        if parameters['use_pos']:
            if self.verbose:
                print('pos_tag_vector_indices[\'train\'][0:2]: {0}'.format(pos_tag_vector_indices['train'][0:2]))
            if self.verbose:
                print('len(pos_tag_vector_indices[\'train\']): {0}'.format(len(pos_tag_vector_indices['train'])))

        if parameters['use_gaz']:
            if self.verbose:
                print('gaz_vector_indices[\'train\'][0:2]: {0}'.format(gaz_vector_indices['train'][0:2]))
            if self.verbose:
                print('len(gaz_vector_indices[\'train\']): {0}'.format(len(gaz_vector_indices['train'])))

        if parameters['use_aff']:
            if self.verbose:
                print('aff_vector_indices[\'train\'][0:2]: {0}'.format(aff_vector_indices['train'][0:2]))
            if self.verbose:
                print('len(aff_vector_indices[\'train\']): {0}'.format(len(aff_vector_indices['train'])))

        if not parameters['use_pos']:
            pos_tag_indices = None
            pos_tag_vector_indices = None
        if not parameters['use_gaz']:
            gaz_indices = None
            gaz_vector_indices = None
        if not parameters['use_aff']:
            aff_indices = None
            aff_vector_indices = None
        return token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices, pos_tag_indices, pos_tag_vector_indices, gaz_indices, gaz_vector_indices, aff_indices, aff_vector_indices

    def update_dataset(self, dataset_filepaths, dataset_types):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        Overwrites the data of type specified in dataset_types using the existing token_to_index, character_to_index, and label_to_index mappings. 
        '''
        for dataset_type in dataset_types:
            self.labels[dataset_type], self.tokens[dataset_type], _, _, _,self.pos_tags[dataset_type] = self._parse_dataset(dataset_filepaths.get(dataset_type, None),parameters)
        
        token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices, pos_tag_indices, pos_tag_vector_indices, gaz_indices, gaz_vector_indices, aff_indices, aff_vector_indices = self._convert_to_indices(dataset_types,parameters)
        
        self.token_indices.update(token_indices)
        self.label_indices.update(label_indices)
        self.character_indices_padded.update(character_indices_padded)
        self.character_indices.update(character_indices)
        self.token_lengths.update(token_lengths)
        self.characters.update(characters)
        self.label_vector_indices.update(label_vector_indices)
        if parameters['use_pos']:
            self.pos_tag_indices.update(pos_tag_indices)
            self.pos_tag_vector_indices.update(pos_tag_vector_indices)
        if parameters['use_gaz']:
            self.gaz_indices.update(gaz_indices)
            self.gaz_vector_indices.update(gaz_vector_indices)
        if parameters['use_aff']:
            self.aff_indices.update(aff_indices)
            self.aff_vector_indices.update(aff_vector_indices)

    def load_dataset(self, dataset_filepaths, parameters, token_to_vector=None):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)
        if parameters['token_pretrained_embedding_filepath'] != '':
            if token_to_vector==None:
                token_to_vector = utils_nlp.load_pretrained_token_embeddings(parameters)
        else:
            token_to_vector = {}
        if self.verbose: print("len(token_to_vector): {0}".format(len(token_to_vector)))

        if parameters['use_gaz']:
            self._parse_gaz(parameters['gaz_filepath'])
        if parameters['use_aff']:
            self._parse_aff(parameters['aff_filepath'])

        # Load pretraining dataset to ensure that index to label is compatible to the pretrained model,
        #   and that token embeddings that are learned in the pretrained model are loaded properly.
        all_tokens_in_pretraining_dataset = []
        all_characters_in_pretraining_dataset = []
        if parameters['use_pretrained_model']:
            pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()
            all_characters_in_pretraining_dataset = pretraining_dataset.index_to_character.values()

        remap_to_unk_count_threshold = 1
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = 'UNK'
        self.unique_labels = []
        labels = {}
        tokens = {}
        pos_tags = {}
        gazs = {}
        affs = {}
        label_count = {}
        token_count = {}
        character_count = {}
        pos_tag_count = {}
        gaz_count = {}
        aff_count = {}
        for dataset_type in ['train', 'valid', 'test', 'deploy']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type], character_count[dataset_type], \
                pos_tags[dataset_type], pos_tag_count[dataset_type],gazs[dataset_type],gaz_count[dataset_type], affs[dataset_type],aff_count[dataset_type] = self._parse_dataset(dataset_filepaths.get(dataset_type, None),parameters)

            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))
        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(token_count['test'].keys()) + list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][token] + token_count['deploy'][token]
        
        if parameters['load_all_pretrained_token_embeddings']:
            for token in token_to_vector:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1
            for token in all_tokens_in_pretraining_dataset:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(character_count['test'].keys()) + list(character_count['deploy'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][character] + character_count['test'][character] + character_count['deploy'][character]

        for character in all_characters_in_pretraining_dataset:
            if character not in character_count['all']:
                character_count['all'][character] = -1
                character_count['train'][character] = -1

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(label_count['test'].keys()) + list(label_count['deploy'].keys()):
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + label_count['test'][character] + label_count['deploy'][character]

        if parameters['use_pos']:
            pos_tag_count['all'] = {}
            for pos_tag in list(pos_tag_count['train'].keys()) + list(pos_tag_count['valid'].keys()) + list(pos_tag_count['test'].keys()) + list(pos_tag_count['deploy'].keys()):
                pos_tag_count['all'][pos_tag] = pos_tag_count['train'][pos_tag] + pos_tag_count['valid'][pos_tag] + pos_tag_count['test'][pos_tag] + pos_tag_count['deploy'][pos_tag]


        if parameters['use_gaz']:
            gaz_count['all'] = {}
            for gaz in list(gaz_count['train'].keys()) + list(gaz_count['valid'].keys()) + list(gaz_count['test'].keys()) + list(gaz_count['deploy'].keys()):
                gaz_count['all'][gaz] = gaz_count['train'][gaz] + gaz_count['valid'][gaz] + gaz_count['test'][gaz] + gaz_count['deploy'][gaz]

        if parameters['use_aff']:
            aff_count['all'] = {}
            for aff in list(aff_count['train'].keys()) + list(aff_count['valid'].keys()) + list(aff_count['test'].keys()) + list(aff_count['deploy'].keys()):
                aff_count['all'][aff] = aff_count['train'][aff] + aff_count['valid'][aff] + aff_count['test'][aff] + aff_count['deploy'][aff]
            
        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse = True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse = False)
        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse = True)
        if parameters['use_pos']:
            pos_tag_count['all'] = utils.order_dictionary(pos_tag_count['all'], 'key', reverse = False)
        if parameters['use_gaz']:
            gaz_count['all'] = utils.order_dictionary(gaz_count['all'], 'key', reverse = False)
        if parameters['use_aff']:
            aff_count['all'] = utils.order_dictionary(aff_count['all'], 'key', reverse = False)
        if self.verbose: print('character_count[\'all\']: {0}'.format(character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        if self.verbose: print("parameters['remap_unknown_tokens_to_unk']: {0}".format(parameters['remap_unknown_tokens_to_unk']))
        if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1

            if parameters['remap_unknown_tokens_to_unk'] == 1 and \
                (token_count['train'][token] == 0 or \
                parameters['load_only_pretrained_token_embeddings']) and \
                not utils_nlp.is_token_in_pretrained_embeddings(token, token_to_vector, parameters) and \
                token not in all_tokens_in_pretraining_dataset:
                if self.verbose: print("token: {0}".format(token))
                if self.verbose: print("token.lower(): {0}".format(token.lower()))
                if self.verbose: print("re.sub('\d', '0', token.lower()): {0}".format(re.sub('\d', '0', token.lower())))
                token_to_index[token] =  self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1
        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        # Ensure that both B- and I- versions exist for each label
        labels_without_bio = set()
        for label in label_count['all'].keys():
            new_label = utils_nlp.remove_bio_from_label_name(label)
            labels_without_bio.add(new_label)
        for label in labels_without_bio:
            if label == 'O':
                continue
            if parameters['tagging_format'] == 'bioes':
                prefixes = ['B-', 'I-', 'E-', 'S-']
            else:
                prefixes = ['B-', 'I-']
            for prefix in prefixes:
                l = prefix + label
                if l not in label_count['all']:
                    label_count['all'][l] = 0
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse = False)

        if parameters['use_pretrained_model']:
            self.unique_labels = sorted(list(pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        if parameters['use_pos']:
            pos_tag_to_index = {}
            iteration_number = 0
            for pos_tag, count in pos_tag_count['all'].items():
                pos_tag_to_index[pos_tag] = iteration_number
                iteration_number += 1

        if parameters['use_gaz']:
            gaz_to_index = {}
            iteration_number = 0
            for gaz, count in gaz_count['all'].items():
                gaz_to_index[gaz] = iteration_number
                iteration_number += 1

        if parameters['use_aff']:
            aff_to_index = {}
            iteration_number = 0
            for aff, count in aff_count['all'].items():
                aff_to_index[aff] = iteration_number
                iteration_number += 1

        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))
        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse = False)
        if self.verbose: print('token_to_index: {0}'.format(token_to_index))
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        if self.verbose: print('index_to_token: {0}'.format(index_to_token))

        if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))
        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse = False)
        if self.verbose: print('label_to_index: {0}'.format(label_to_index))
        index_to_label = utils.reverse_dictionary(label_to_index)
        if self.verbose: print('index_to_label: {0}'.format(index_to_label))

        character_to_index = utils.order_dictionary(character_to_index, 'value', reverse = False)
        index_to_character = utils.reverse_dictionary(character_to_index)
        if self.verbose: print('character_to_index: {0}'.format(character_to_index))
        if self.verbose: print('index_to_character: {0}'.format(index_to_character))

        if parameters['use_pos']:
            if self.verbose: print('pos_tag_count[\'train\']: {0}'.format(pos_tag_count['train']))
            pos_tag_to_index = utils.order_dictionary(pos_tag_to_index, 'value', reverse = False)
            if self.verbose: print('pos_tag_to_index: {0}'.format(pos_tag_to_index))
            index_to_pos_tag = utils.reverse_dictionary(pos_tag_to_index)
            if self.verbose: print('index_to_pos_tag: {0}'.format(index_to_pos_tag))

        if parameters['use_gaz']:
            if self.verbose: print('gaz_count[\'train\']: {0}'.format(gaz_count['train']))
            gaz_to_index = utils.order_dictionary(gaz_to_index, 'value', reverse = False)
            if self.verbose: print('gaz_to_index: {0}'.format(gaz_to_index))
            index_to_gaz = utils.reverse_dictionary(gaz_to_index)
            if self.verbose: print('index_to_gaz: {0}'.format(index_to_gaz))

        if parameters['use_aff']:
            if self.verbose: print('aff_count[\'train\']: {0}'.format(aff_count['train']))
            aff_to_index = utils.order_dictionary(aff_to_index, 'value', reverse = False)
            if self.verbose: print('aff_to_index: {0}'.format(aff_to_index))
            index_to_aff = utils.reverse_dictionary(aff_to_index)
            if self.verbose: print('index_to_aff: {0}'.format(index_to_aff))


        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        if self.verbose:
            # Print sequences of length 1 in train set
            for token_sequence, label_sequence in zip(tokens['train'], labels['train']):
                if len(label_sequence) == 1 and label_sequence[0] != 'O':
                    print("{0}\t{1}".format(token_sequence[0], label_sequence[0]))

        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        if parameters['use_pos']:
            self.index_to_pos_tag = index_to_pos_tag
            self.pos_tag_to_index = pos_tag_to_index
        if parameters['use_gaz']:
            self.index_to_gaz = index_to_gaz
            self.gaz_to_index = gaz_to_index
        if parameters['use_aff']:
            self.index_to_aff = index_to_aff
            self.aff_to_index = aff_to_index
        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))
        self.tokens = tokens
        self.labels = labels
        if parameters['use_pos']:
            self.pos_tags = pos_tags
        if parameters['use_gaz']:
            self.gazs = gazs
        if parameters['use_aff']:
            self.affs = affs

        token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices, pos_tag_indices, pos_tag_vector_indices, gaz_indices, gaz_vector_indices, aff_indices, aff_vector_indices = self._convert_to_indices(dataset_filepaths.keys(),parameters)
        
        self.token_indices = token_indices
        self.label_indices = label_indices
        self.character_indices_padded = character_indices_padded
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.characters = characters
        self.label_vector_indices = label_vector_indices
        if parameters['use_pos']:
            self.pos_tag_indices = pos_tag_indices
            self.pos_tag_vector_indices = pos_tag_vector_indices
        if parameters['use_gaz']:
            self.gaz_indices = gaz_indices
            self.gaz_vector_indices = gaz_vector_indices
        if parameters['use_aff']:
            self.aff_indices = aff_indices
            self.aff_vector_indices = aff_vector_indices

        self.number_of_classes = max(self.index_to_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.alphabet_size = max(self.index_to_character.keys()) + 1
        if parameters['use_pos']:
            self.number_of_POS_types = max(self.index_to_pos_tag.keys()) + 1
        if self.verbose: print("self.number_of_classes: {0}".format(self.number_of_classes))
        if self.verbose: print("self.alphabet_size: {0}".format(self.alphabet_size))
        if self.verbose: print("self.vocabulary_size: {0}".format(self.vocabulary_size))
        if parameters['use_pos']:
            if self.verbose: print("self.number_of_POS_types: {0}".format(self.number_of_POS_types))

        # unique_labels_of_interest is used to compute F1-scores.
        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')

        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])

        self.infrequent_token_indices = infrequent_token_indices

        if self.verbose: print('self.unique_labels_of_interest: {0}'.format(self.unique_labels_of_interest))
        if self.verbose: print('self.unique_label_indices_of_interest: {0}'.format(self.unique_label_indices_of_interest))

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        
        return token_to_vector

