# ----------------------------------------------------------------------------------------------
# Import dependencies
# ----------------------------------------------------------------------------------------------

from settings import *

import sys
import math
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import os
import numpy as np
import _pickle as pickle
import time
import csv
from collections import defaultdict

from tensorflow.keras.models import load_model, model_from_yaml
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tikzplotlib import save as tikz_save
import pretty_midi as pm
import scipy

import midi_functions as mf
import vae_definition
from vae_definition import VAE
from vae_definition import KLDivergenceLayer
import data_class
from import_midi import import_midi_from_folder


# ----------------------------------------------------------------------------------------------
# Set schedule for the evaluation
# ----------------------------------------------------------------------------------------------

harmonicity_evaluations = True
frankenstein_harmonicity_evaluations = True # runs only if harmonicity_evaluations are turned on

max_new_chosen_interpolation_songs = 3 #42
interpolation_length = 4 #how many iterations?
how_many_songs_in_one_medley = 3
noninterpolated_samples_between_interpolation = 8 #should be at least 1, otherwise it can not interpolate

max_new_sampled_interpolation_songs = 3 #42
interpolation_song_length = 16 #how many iterations?

latent_sweep = True
num_latent_sweep_samples = 3 #100
num_latent_sweep_evaluation_songs = 10

chord_evaluation = True
evaluate_different_sampling_regions = True
pitch_evaluation = True
max_new_sampled_songs = 3 #100
max_new_sampled_long_songs = 3 #100

evaluate_autoencoding_and_stuff = True
mix_with_previous = True
switch_styles = True


# ----------------------------------------------------------------------------------------------
# Model library (Change those strings to use it)
# ----------------------------------------------------------------------------------------------


model_name = '20220310-104742-_ls_inlen_64_outlen_64_beta_0.1_lr_0.0002_lstmsize_256_latent_256_trainsize_815_testsize_91_epsstd_0.01/'
epoch = 90




if test_train_set:
    set_string = 'train/'
else:
    set_string = 'test/'

model_path = 'models/autoencode/vae/' + model_name
save_folder = 'autoencode_midi/vae_eval/' + model_name[:10] + '/' + set_string

 


if not os.path.exists(save_folder):
    os.makedirs(save_folder)   



# ----------------------------------------------------------------------------------------------
# Evaluation settings
# ----------------------------------------------------------------------------------------------

model_filetype = '.pickle'


max_plots_per_song = 3

BPM = 100

shuffle = False
composer_decoder_latent_size = 10

assert(output_length > 0)

verbose = False

sample_method = 'argmax' #choice, argmax

# ----------------------------------------------------------------------------------------------
# Import and preprocess data
# ----------------------------------------------------------------------------------------------

print('loading data...')
# Get Train and test sets

folder = source_folder

V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, C_train, C_test, train_paths, test_paths = import_midi_from_folder(folder)

train_set_size = len(X_train)
test_set_size = len(X_test)

# ----------------------------------------------------------------------------------------------
# Simple statistics on train and test set
# ----------------------------------------------------------------------------------------------


total_train_songs_per_class = [0 for _ in range(num_classes)]
total_train_samples_per_class = [0 for _ in range(num_classes)]

total_test_songs_per_class = [0 for _ in range(num_classes)]
total_test_samples_per_class = [0 for _ in range(num_classes)]

for i, C in enumerate(C_train):
    total_train_songs_per_class[C] += 1
    total_train_samples_per_class[C] += X_train[i].shape[0]

for i, C in enumerate(C_test):
    total_test_songs_per_class[C] += 1
    total_test_samples_per_class[C] += X_test[i].shape[0]

print("Total train songs per class: ", total_train_songs_per_class)
print("Total train samples per class: ", total_train_samples_per_class)
print("Total test songs per class: ", total_test_songs_per_class)
print("Total test samples per class: ", total_test_samples_per_class)

print("Classes", classes)
print("Model name", model_name)
print("Test on train set", test_train_set)
input("Correct settings?")

# ----------------------------------------------------------------------------------------------
# Instruments (midi programs) statistics
# ----------------------------------------------------------------------------------------------


programs_for_each_class = [[] for _ in range(num_classes)]
for train_song_num in range(len(Y_train)):
    C = C_train[train_song_num]
    I = I_train[train_song_num]
    programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
    for program in programs:
        if not program in programs_for_each_class[C]:
            programs_for_each_class[C].append(program)

print(programs_for_each_class)


#calculate how many programs have to be switched on average for a style change on the training set
all_programs_plus_length_for_each_class = [[] for _ in range(num_classes)]
total_programs_for_each_class = [0 for _ in range(num_classes)]
program_probability_dict_for_each_class = [dict() for _ in range(num_classes)]
for i in range(len(I_train)):
    num_samples = X_train[i].shape[0] #get the number of samples to know how many splitted songs there are for this original song
    I = I_train[i]
    C = C_train[i]
    programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
    all_programs_plus_length_for_each_class[C].append((programs, num_samples))
    total_programs_for_each_class[C] += num_samples * max_voices
    for program in programs:
        program_probability_dict_for_each_class[C][program] = program_probability_dict_for_each_class[C].get(program, 0) + num_samples

for d in program_probability_dict_for_each_class:
    print(d)

#divide by total number of programs to get a probability for each key
for C, d in enumerate(program_probability_dict_for_each_class):
    for k in d.keys():
        d[k] /= total_programs_for_each_class[C]
            

for d in program_probability_dict_for_each_class:
    print(d)

#enlist the possible instruments for each class
if instrument_attach_method == '1hot-category' or 'khot-category':
    possible_programs = list(range(0,127,8))
else:
    possible_programs = list(range(0,127))

#calculate the random probability for each class
print("Calculate how probable your instrument picks are if you pick them completely random: ")
for C, class_name in enumerate(classes):
    probabilities_for_this_class = []
    for program in possible_programs:
        probabilities_for_this_class.append(program_probability_dict_for_each_class[C].get(program, 0))
    print("Random probability for class " + class_name + ": ", np.mean(probabilities_for_this_class))
    #of course, this is the same as 1/len(possible_programs)


#calculate the instrument probability for each class
print("Calculate how probable your instrument picks are if you don't switch any instrument and stay in the same class: ")
for C, class_name in enumerate(classes):
    probability_for_this_class = 0
    for (programs, length) in all_programs_plus_length_for_each_class[C]:
        for program in programs:
            probability_for_this_class += length * program_probability_dict_for_each_class[C].get(program, 0)
    probability_for_this_class /= total_programs_for_each_class[C]
    print("Same probability for class " + class_name + ": ", probability_for_this_class)


#calculate the instrument probability for each class
print("Calculate how probable your instrument picks are in another class if you don't switch any instrument: ")
for C, class_name in enumerate(classes):
    
    for C_switch, class_name_switch in enumerate(classes):
        if C != C_switch:
            probability_for_other_class = 0
            for (programs, length) in all_programs_plus_length_for_each_class[C]:
                for program in programs:
                    probability_for_other_class += length * program_probability_dict_for_each_class[C_switch].get(program, 0)
            probability_for_other_class /= total_programs_for_each_class[C]
            print("Probability that a program-pick from class " + class_name + " is occuring class " + class_name_switch +" : ", probability_for_other_class)

for C, class_name in enumerate(classes):
    programs_plus_length_for_this_class = all_programs_plus_length_for_each_class[C]
    print(len(programs_plus_length_for_this_class))
    for C_switch, class_name_switch in enumerate(classes):
        if C_switch != C:
            print("Calculating how many instruments switches have to be made from " + class_name + " to " + class_name_switch)
            same = 0.0
            different = 0.0
            programs_plus_length_for_other_class = all_programs_plus_length_for_each_class[C_switch]
            for programs, length in programs_plus_length_for_this_class:
                for programs_switch, length_switch in programs_plus_length_for_other_class:
                    for this_program, other_program in zip(programs, programs_switch):
                        if this_program == other_program:
                            same += length * length_switch
                        else:
                            different += length * length_switch
            print("Switch percentage: ", different / (same + different))


# ----------------------------------------------------------------------------------------------
# Prepare signature vectors
# ----------------------------------------------------------------------------------------------

S_train_for_each_class = [[] for _ in range(num_classes)]
S_test_for_each_class = [[] for _ in range(num_classes)]
all_S = []
S_train = []
for train_song_num in range(len(Y_train)):
    Y = Y_train[train_song_num]
    C = C_train[train_song_num]
    num_samples = Y.shape[0]
    signature_vectors = np.zeros((num_samples, signature_vector_length))
    for sample in range(num_samples):
        poly_sample = data_class.monophonic_to_khot_pianoroll(Y[sample], max_voices)
        if include_silent_note:
            poly_sample = poly_sample[:,:-1]
        signature = data_class.signature_from_pianoroll(poly_sample)
        signature_vectors[sample] = signature
    S_train.append(signature_vectors)
    all_S.extend(signature_vectors)
    S_train_for_each_class[C].extend(signature_vectors)

all_S = np.asarray(all_S)

mean_signature = np.mean(all_S, axis=0)
print(mean_signature)
std_signature = np.std(all_S, axis=0)

#make sure you don't divide by zero if std is 0
for i, val in enumerate(std_signature):
    if val == 0:
        std_signature[i] = 1.0e-10
print(std_signature)


normalized_S_train = []
for signature_vectors in S_train:
    normalized_signature_vectors = (signature_vectors - mean_signature) / std_signature
    normalized_S_train.append(normalized_signature_vectors)

normalized_S_test = []
S_test = []
for test_song_num in range(len(Y_test)):
    Y = Y_test[test_song_num]
    C = C_test[test_song_num]
    num_samples = Y.shape[0]
    signature_vectors = np.zeros((num_samples, signature_vector_length))
    normalized_signature_vectors = np.zeros((num_samples, signature_vector_length))
    for sample in range(num_samples):
        poly_sample = data_class.monophonic_to_khot_pianoroll(Y[sample], max_voices)
        if include_silent_note:
            poly_sample = poly_sample[:,:-1]
        signature = data_class.signature_from_pianoroll(poly_sample)
        normalized_signature_vectors[sample] = signature
        signature = (signature - mean_signature) / std_signature
        normalized_signature_vectors[sample] = signature
    normalized_S_test.append(signature_vectors)
    S_test_for_each_class[C].extend(signature_vectors)
    S_test.append(signature_vectors)


normalized_S_test = np.asarray(normalized_S_test)
S_test = np.asarray(S_test)

normalized_S_train = np.asarray(normalized_S_train)
S_test = np.asarray(S_train)

S_train_for_each_class = np.asarray(S_train_for_each_class)
S_test_for_each_class = np.asarray(S_test_for_each_class)


# ----------------------------------------------------------------------------------------------
# Build VAE and load from weights
# ----------------------------------------------------------------------------------------------

#You have to create the model again with the same parameters as in training and set the weights manually
#There is an issue with storing the model with the recurrentshop extension

if do_not_sample_in_evaluation:
    e = 0.0
else:
    e = epsilon_std


model = VAE()
model.create( input_dim=input_dim, 
    output_dim=output_dim, 
    use_embedding=use_embedding, 
    embedding_dim=embedding_dim, 
    input_length=input_length,
    output_length=output_length, 
    latent_rep_size=latent_dim, 
    vae_loss=vae_loss,
    optimizer=optimizer, 
    activation=activation, 
    lstm_activation=lstm_activation, 
    lstm_state_activation=lstm_state_activation,
    epsilon_std=e, 
    epsilon_factor=epsilon_factor,
    include_composer_decoder=include_composer_decoder,
    num_composers=num_composers, 
    composer_weight=composer_weight, 
    lstm_size=lstm_size, 
    cell_type=cell_type,
    num_layers_encoder=num_layers_encoder, 
    num_layers_decoder=num_layers_decoder, 
    bidirectional=bidirectional, 
    decode=decode, 
    teacher_force=teacher_force, 
    learning_rate=learning_rate, 
    split_lstm_vector=split_lstm_vector, 
    history=history, 
    beta=beta, 
    prior_mean=prior_mean,
    prior_std=prior_std,
    decoder_additional_input=decoder_additional_input, 
    decoder_additional_input_dim=decoder_additional_input_dim, 
    extra_layer=extra_layer,
    meta_instrument= meta_instrument,
    meta_instrument_dim= meta_instrument_dim,
    meta_instrument_length=meta_instrument_length,
    meta_instrument_activation=meta_instrument_activation,
    meta_instrument_weight = meta_instrument_weight,
    signature_decoder = signature_decoder,
    signature_dim = signature_dim,
    signature_activation = signature_activation,
    signature_weight = signature_weight,
    composer_decoder_at_notes_output=composer_decoder_at_notes_output,
    composer_decoder_at_notes_weight=composer_decoder_at_notes_weight,
    composer_decoder_at_notes_activation=composer_decoder_at_notes_activation,
    composer_decoder_at_instrument_output=composer_decoder_at_instrument_output,
    composer_decoder_at_instrument_weight=composer_decoder_at_instrument_weight,
    composer_decoder_at_instrument_activation=composer_decoder_at_instrument_activation,
    meta_velocity=meta_velocity,
    meta_velocity_length=meta_velocity_length,
    meta_velocity_activation=meta_velocity_activation,
    meta_velocity_weight=meta_velocity_weight,
    meta_held_notes=meta_held_notes,
    meta_held_notes_length=meta_held_notes_length,
    meta_held_notes_activation=meta_held_notes_activation,
    meta_held_notes_weight=meta_held_notes_weight,
    meta_next_notes=meta_next_notes,
    meta_next_notes_output_length=meta_next_notes_output_length,
    meta_next_notes_weight=meta_next_notes_weight,
    meta_next_notes_teacher_force=meta_next_notes_teacher_force,
    activation_before_splitting=activation_before_splitting
    )

autoencoder = model.autoencoder
autoencoder.load_weights(model_path+'autoencoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)

encoder = model.encoder
encoder.load_weights(model_path+'encoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)

decoder = model.decoder
decoder.load_weights(model_path+'decoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)


print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

if reset_states:
    autoencoder.reset_states()
    encoder.reset_states()
    decoder.reset_states()


# ----------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------


#spherical linear interpolation
def slerp(p0, p1, t):
    omega = arccos(dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = sin(omega)
    return sin((1.0-t)*omega) / so * p0 + sin(t*omega)/so * p1

def linear_interpolation(p0, p1, t):
    return p0 * (1.0-t) + p1 * t


def split_song_back_to_samples(X, length):
    number_of_splits = int(X.shape[0] / length)
    splitted_songs = np.split(X, number_of_splits)
    return splitted_songs

#I_pred instrument prediction of shape (num_samples, max_voices, different_instruments)
#returns list of program numbers of length max_voices
def vote_for_programs(I_pred):
    program_voting_dict_for_each_voice = [dict() for _ in range(max_voices)]
    for instrument_feature_matrix in I_pred:
        programs = data_class.instrument_representation_to_programs(instrument_feature_matrix, instrument_attach_method)

        for voice, program in enumerate(programs):
            program_voting_dict_for_each_voice[voice][program] = program_voting_dict_for_each_voice[voice].get(program,0) + 1

    #determine mixed_programs_for_whole_song by taking the instruments for each track with the most occurence in the mixed predictions
    programs_for_whole_long_song = []
    for voice in range(max_voices):
        best_program = 0
        highest_value = 0
        for k in program_voting_dict_for_each_voice[voice].keys():
            if program_voting_dict_for_each_voice[voice][k] > highest_value:
                best_program = k 
                highest_value = program_voting_dict_for_each_voice[voice][k]
        programs_for_whole_long_song.append(best_program)

    return programs_for_whole_long_song

def prepare_for_drawing(Y, V=None):
    #use V to make a grey note if it is more silent
    newY = np.copy(Y)
    if V is not None:
        for step in range(V.shape[0]):
            
            if V[step] > velocity_threshold_such_that_it_is_a_played_note:
                velocity = (V[step] - velocity_threshold_such_that_it_is_a_played_note) * MAX_VELOCITY
                newY[step,:] *= velocity
            else:
                if step > max_voices:
                    previous_pitch = np.argmax(newY[step-max_voices])
                    current_pitch = np.argmax(newY[step])
                    if current_pitch != previous_pitch:
                        newY[step,:] = 0
                    else:
                        newY[step,:] = newY[step-max_voices,:]
                else:
                    newY[step,:] = 0

        Y_poly = data_class.monophonic_to_khot_pianoroll(newY, max_voices, set_all_nonzero_to_1=False)
    else:
        Y_poly = data_class.monophonic_to_khot_pianoroll(newY, max_voices)
    return np.transpose(Y_poly)


def restructure_song_to_fit_more_instruments(Y, I_list, V, D):

    num_samples = len(I_list)
    Y_final = np.zeros((num_samples * output_length * num_samples, Y.shape[1]))
    V_final = np.zeros((num_samples * output_length * num_samples,))
    D_final = np.zeros((num_samples * output_length * num_samples,))
    final_programs = []
    for sample, I in enumerate(I_list):
        programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
        final_programs.extend(programs)

        
        for step in range(output_length//max_voices):
            for voice in range(max_voices):
                Y_final[sample * output_length * num_samples + step * num_samples * max_voices + voice,:] = Y[sample *output_length+ step*max_voices + voice,:]
                V_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = V[sample *output_length+ step*max_voices + voice]
                D_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = D[sample *output_length + step*max_voices + voice]
    return Y_final, final_programs, V_final, D_final


# ----------------------------------------------------------------------------------------------
# Save latent train lists
# ----------------------------------------------------------------------------------------------

print("Saving latent train lists...")



train_representation_list = []
all_z = []
for train_song_num in range(len(X_train)):

    #create dataset
    song_name = train_paths[train_song_num].split('/')[-1]
    song_name = song_name.replace('mid.pickle', '')
    X = X_train[train_song_num]
    C = C_train[train_song_num] 
    I = I_train[train_song_num]
    V = V_train[train_song_num]
    D = D_train[train_song_num]

    encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
    #get the latent representation of every song part
    encoded_representation = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
    train_representation_list.append(encoded_representation)
    all_z.extend(encoded_representation)
    train_save_folder = save_folder
    if not test_train_set:
        train_save_folder = save_folder[:-5] + 'train/'
    if not os.path.exists(train_save_folder+ classes[C]+'/'):
        os.makedirs(train_save_folder + classes[C]+'/') 
    if save_anything: np.save(train_save_folder + classes[C]+'/'+'z_' + song_name, encoded_representation)

z_mean_train = np.mean(np.asarray(all_z))
z_std_train = np.std(np.asarray(all_z))

print("z mean train: ", z_mean_train)
print("z std train: ", z_std_train)


# ----------------------------------------------------------------------------------------------
# Generation of random interpolation songs
# ----------------------------------------------------------------------------------------------

sample_method = 'argmax'

for song_num in range(max_new_sampled_interpolation_songs):

    print("Producing random interpolation song ", song_num)

    random_code_1 = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
    random_code_2 = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))

    C = 1

    Y_list = []
    V_list = []
    D_list = []
    I_list = []
    previous_latent_rep = np.zeros((1,latent_dim))
    S = np.zeros((1, signature_vector_length))

    
    for i in range(interpolation_song_length+1):
        R = linear_interpolation(random_code_1, random_code_2, i/float(interpolation_song_length))
        interpolation_input_list = vae_definition.prepare_decoder_input(R, C, S, previous_latent_rep)
        decoder_outputs = decoder.predict(interpolation_input_list, batch_size=batch_size, verbose=False)
        Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
        Y_list.extend(Y)
        I_list.extend(I)
        V_list.extend(V)
        D_list.extend(D)

        previous_latent_rep = R

    programs_for_whole_long_song = vote_for_programs(I_list)

    Y_list = np.asarray(Y_list)
    D_list = np.asarray(D_list)
    V_list = np.asarray(V_list)



    if save_anything: data_class.draw_pianoroll(prepare_for_drawing(Y_list, V_list), name='random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length), show=False, save_path=save_folder +'random_interpolation_' + str(song_num)+'_length_' + str(interpolation_song_length))
    if save_anything: mf.rolls_to_midi(Y_list, programs_for_whole_long_song, save_folder, 'random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length), BPM, V_list, D_list)
    Y_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(Y_list, I_list, V_list, D_list)
    if save_anything: mf.rolls_to_midi(Y_all_programs, all_programs, save_folder, 'random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length) + '_all_programs', BPM, V_all_programs, D_all_programs)

# ----------------------------------------------------------------------------------------------
# Generation of new song parts
# ----------------------------------------------------------------------------------------------
sample_method = 'choice'

for song_num in range(max_new_sampled_songs):

    #prepare random decoder input list
    C = 1
    S = np.zeros((1, signature_vector_length))
    R = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
    random_input_list = vae_definition.prepare_decoder_input(R, C, S)

    decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

    Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
    
    programs = data_class.instrument_representation_to_programs(I[0], instrument_attach_method)

    if save_anything: mf.rolls_to_midi(Y, programs, save_folder, 'random_'+str(song_num), BPM, V, D)

    if include_composer_decoder:

        previous_song = None
        previous_programs = None

        random_code = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
        for C in range(num_classes):
            
            #turn the knob to one class:
            random_code[0,0:num_classes] = -1
            random_code[0,C] = 1

            S = np.zeros((1, signature_vector_length))
            R = random_code
            random_input_list = vae_definition.prepare_decoder_input(R, C, S)
            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

            programs = data_class.instrument_representation_to_programs(I[0], instrument_attach_method)

            if previous_song is not None:
                data_class.draw_difference_pianoroll(prepare_for_drawing(Y), prepare_for_drawing(previous_song), name_1=str(song_num) +"_" + str(C) + " Programs: " + str(programs) , name_2=str(song_num) +"_" + str(C-1) + " Programs: " + str(previous_programs), show=False, save_path=save_folder+"random_"+str(song_num) +"_" +str(C)+ "_vs_" + str(C-1) +"_switchdiff.png")

            if save_anything: mf.rolls_to_midi(Y, programs, save_folder, 'random_'+str(song_num) + "_" + str(C), BPM, V, D)

            previous_song = Y
            previous_programs = programs

            
# ----------------------------------------------------------------------------------------------
# Generation of new long songs
# ----------------------------------------------------------------------------------------------


long_song_length = 20 #how many iterations?

for song_num in range(max_new_sampled_long_songs):

    print("Producing random song ", song_num)

    if include_composer_decoder:

        random_code = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))

        C = 1 #content class
        R = random_code #latent space

        Y_list = [] #pitch
        V_list = [] #velocity
        D_list = [] #duration
        I_list = [] #instrument 
        previous_latent_rep = np.zeros((1,latent_dim))

        S = np.zeros((1, signature_vector_length))

        already_picked_z_indices = []
        
        for i in range(long_song_length):


            lowest_distance = np.linalg.norm(all_z[0]-R)  
            best_z_index = 0
            for i, z in enumerate(all_z):
                distance = np.linalg.norm(z-R)
                if distance < lowest_distance and i not in already_picked_z_indices:
                    lowest_distance = distance
                    best_z_index = i

            already_picked_z_indices.append(best_z_index)
            closest_z = all_z[best_z_index]
            print("Closest z index : ", best_z_index)

            #e = np.random.normal(loc=0.0, scale=1.0, size=(1,latent_dim))
            e = np.random.rand()
            e = z_std_train
            R = (R + closest_z * e) / (1 + e)

            

            random_input_list = vae_definition.prepare_decoder_input(R, C, S, previous_latent_rep)
            
            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

            Y_list.extend(Y)
            I_list.extend(I)
            V_list.extend(V)
            D_list.extend(D)

            #use output as next input
            X = np.copy(Y)
            if include_silent_note:
                X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)
                for step in range(X.shape[0]):
                    if np.sum(X[step]) == 0:
                        X[step, -1] = 1
            X = np.asarray([X])

            previous_latent_rep = R
            encoder_input_list = vae_definition.prepare_encoder_input_list(X,I[0],np.expand_dims(V, axis=0),np.expand_dims(D,axis=0))
            R = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)

        programs_for_whole_long_song = vote_for_programs(I_list)

        Y_list = np.asarray(Y_list)
        D_list = np.asarray(D_list)
        V_list = np.asarray(V_list)

        if save_anything: mf.rolls_to_midi(Y_list, programs_for_whole_long_song, save_folder, 'random_long_temp' + str(temperature) + "_" + str(song_num), BPM, V_list, D_list)
