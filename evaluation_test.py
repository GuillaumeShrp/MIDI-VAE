
from enum import auto
from import_midi import import_midi_from_folder, import_midi_solo
from settings import *
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import time
import midi_functions as mf
import vae_definition
from vae_definition import VAE
import data_class
import tensorflow as tf 
from keras.utils import to_categorical

# remove depreciation warnings 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ----------------------------------------------------------------------------------------------
# Model library
# ----------------------------------------------------------------------------------------------

#model_name = '20220624-061944-_ls_inlen_64_outlen_64_beta_0.1_lr_0.0002_lstmsize_256_latent_256_trainsize_805_testsize_90_epsstd_0.01/' #JSBTGM 90
#model_name = '20220628-161736-_ls_inlen_64_outlen_64_beta_0.1_lr_0.0002_lstmsize_256_latent_256_trainsize_656_testsize_73_epsstd_0.01/' #JZZCLS 100
model_name = '20220701-080237-_ls_inlen_64_outlen_64_beta_0.1_lr_0.0002_lstmsize_256_latent_256_trainsize_656_testsize_73_epsstd_0.01/' #NONE
epoch = 0


model_path = 'models/vae/' + model_name
#save_folder = 'evaluations/vae/' + model_name[:10] + '_JSBNTG_CHECKMIDIROLL/'
save_folder = 'evaluations/midi2roll2midi/'

# ----------------------------------------------------------------------------------------------
# Evaluation settings
# ----------------------------------------------------------------------------------------------

BPM = 100

assert(output_length > 0)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)   

test_reconstr = True
load_model_weights = False
test_identity = False
test_none_generation = False

# ----------------------------------------------------------------------------------------------
# Build VAE and load from weights
# ----------------------------------------------------------------------------------------------

#You have to create the model again with the same parameters as in training and set the weights manually
#There is an issue with storing the model with the recurrentshop extension

model = VAE()
model.create(input_dim=input_dim,output_dim=output_dim,use_embedding=use_embedding,embedding_dim=embedding_dim,input_length=input_length,output_length=output_length, latent_rep_size=latent_dim, vae_loss=vae_loss,optimizer=optimizer, activation=activation, lstm_activation=lstm_activation, lstm_state_activation=lstm_state_activation,epsilon_std=epsilon_std, epsilon_factor=epsilon_factor,include_composer_decoder=include_composer_decoder,num_composers=num_composers, composer_weight=composer_weight, lstm_size=lstm_size, cell_type=cell_type,num_layers_encoder=num_layers_encoder, num_layers_decoder=num_layers_decoder, bidirectional=bidirectional, decode=decode, teacher_force=teacher_force, learning_rate=learning_rate, split_lstm_vector=split_lstm_vector, history=history, beta=beta, prior_mean=prior_mean,prior_std=prior_std,decoder_additional_input=decoder_additional_input, decoder_additional_input_dim=decoder_additional_input_dim, extra_layer=extra_layer,meta_instrument= meta_instrument,meta_instrument_dim= meta_instrument_dim,meta_instrument_length=meta_instrument_length,meta_instrument_activation=meta_instrument_activation,meta_instrument_weight = meta_instrument_weight,signature_decoder = signature_decoder,signature_dim = signature_dim,signature_activation = signature_activation,signature_weight = signature_weight,composer_decoder_at_notes_output=composer_decoder_at_notes_output,composer_decoder_at_notes_weight=composer_decoder_at_notes_weight,composer_decoder_at_notes_activation=composer_decoder_at_notes_activation,composer_decoder_at_instrument_output=composer_decoder_at_instrument_output,composer_decoder_at_instrument_weight=composer_decoder_at_instrument_weight,composer_decoder_at_instrument_activation=composer_decoder_at_instrument_activation,meta_velocity=meta_velocity,meta_velocity_length=meta_velocity_length,meta_velocity_activation=meta_velocity_activation,meta_velocity_weight=meta_velocity_weight,meta_held_notes=meta_held_notes,meta_held_notes_length=meta_held_notes_length,meta_held_notes_activation=meta_held_notes_activation,meta_held_notes_weight=meta_held_notes_weight,meta_next_notes=meta_next_notes,meta_next_notes_output_length=meta_next_notes_output_length,meta_next_notes_weight=meta_next_notes_weight,meta_next_notes_teacher_force=meta_next_notes_teacher_force,activation_before_splitting=activation_before_splitting)


encoder = model.encoder
decoder = model.decoder
autoencoder = model.autoencoder

if load_model_weights:
    encoder.load_weights(model_path+'encoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)
    decoder.load_weights(model_path+'decoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)
    autoencoder.load_weights(model_path+'autoencoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)


# ----------------------------------------------------------------------------------------------
# Usefull functions
# ----------------------------------------------------------------------------------------------


def prepare_for_drawing(Y, V=None):
    #use V to make a grey note if it is more silent
    newY = np.copy(Y)
    if V is not None:
        print("shape of V: ",V.shape)
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



def restructure_song_to_fit_more_instruments(Y, I, V, D):

    num_samples = len(I)
    Y_final = np.zeros((num_samples * output_length * num_samples, Y.shape[1]))
    V_final = np.zeros((num_samples * output_length * num_samples,))
    D_final = np.zeros((num_samples * output_length * num_samples,))
    final_programs = []
    for sample, I in enumerate(I):
        programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
        final_programs.extend(programs)

        
        for step in range(output_length//max_voices):
            for voice in range(max_voices):
                Y_final[sample * output_length * num_samples + step * num_samples * max_voices + voice,:] = Y[sample *output_length+ step*max_voices + voice,:]
                V_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = V[sample *output_length+ step*max_voices + voice]
                D_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = D[sample *output_length + step*max_voices + voice]
    return Y_final, final_programs, V_final, D_final


# ----------------------------------------------------------------------------------------------
# Import and preprocess data
# ----------------------------------------------------------------------------------------------

print('loading data...')
folder = 'data/solo/'
V, D, T, I, Y, X, C, name = import_midi_solo(folder, C=0)

#V, D, T, I, Y, X, C, name = import_midi_from_folder(folder)

# ----------------------------------------------------------------------------------------------
# MIDI --> roll --> MIDI reconstruction check up
# ----------------------------------------------------------------------------------------------

if test_reconstr:
    medley_name = name+str(time.time())[7:10]

    print("y shape",X.shape) 
    X = X[:,:,:-1]
    X = X.reshape(-1, 60)
    print("y shape",X.shape) 
    V = V.reshape(-1,)
    
    D = D.reshape(-1,)
    print("y shape",X.shape) #ok 0
    print("i shape",I.shape) #cf ln 182 : ok
    print("v shape",V.shape) #deniere des inputlist(output de prepare_autoencoder_input_and_output_list)
    print("d shape",D.shape) 

    #X, I, V, D, N = vae_definition.process_decoder_outputs([X, I, V], sample_method) #detruit V et X 
    X = np.asarray(X)
    D = np.asarray(D)
    V = np.asarray(V)

    #data_class.draw_pianoroll(prepare_for_drawing(X, V), name=medley_name, show=True, save_path=save_folder + medley_name)
    X_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(X, I, V, D)
    mf.rolls_to_midi(X_all_programs, all_programs, save_folder, medley_name, BPM, V_all_programs, D_all_programs)


# ----------------------------------------------------------------------------------------------
# Identity through VAE
# ----------------------------------------------------------------------------------------------

if test_identity:
    medley_name = "JSBNTG_RCNSTR_"+name+str(time.time())[7:10]


    S = np.random.rand(signature_vector_length).reshape((1, signature_vector_length))

    #get the latent representation of every song part
    encoder_input_list = vae_definition.prepare_encoder_input_list(X,C,I,V,D)
    encoded_representation = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
    H = np.asarray(encoded_representation)

    print("y shape before preprae_autoencdoe_input ")
    for r in [X,Y,I,V,D,S,H]:
        print("init :",r.shape)
    input_list, output_list = vae_definition.prepare_autoencoder_input_and_output_list(X,Y,C,I,V,D,S,H) # I(4,16) --> I(24,4,16)
    print("y shape after preprae_autoencdoe_input [x,c,y,h,instr_start,instr_input,velo_st,velo_inp]")
    for r in input_list:
        print("input :",r.shape)

    
    autoencoder_outputs = autoencoder.predict(input_list, batch_size=batch_size, verbose=False)
    print("y shape after predict : [Y,I,V,D] ")
    for r in autoencoder_outputs:
        print("output ;",r.shape)
    Y, I, V, D, N = vae_definition.process_autoencoder_outputs(autoencoder_outputs, sample_method) # [Y,I,V,D] X(24,26,61) --> Y(24*64,60)

    Y = np.asarray(Y)
    D = np.asarray(D)
    V = np.asarray(V)
    
    print("y shape",Y.shape)
    print("d shape",D.shape)
    print("v shape",V.shape)

    data_class.draw_pianoroll(prepare_for_drawing(Y, V), name=medley_name, show=False, save_path=save_folder +medley_name)
    Y_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(Y, I, V, D)
    mf.rolls_to_midi(Y_all_programs, all_programs, save_folder, medley_name, BPM, V_all_programs, D_all_programs)


# ----------------------------------------------------------------------------------------------
#  Test generation from latent space on non trained model output
# ----------------------------------------------------------------------------------------------

if test_none_generation:
    for i in range(10):
        song_length = 20
        medley_name = "NONE_"+str(time.time())[7:10]
        Y = []
        V = []
        D = []
        I = []
        C = 0

        for j in range(song_length):
            latent_rep = np.random.rand(latent_dim).reshape((1,latent_dim))
            S = np.random.rand(signature_vector_length).reshape((1, signature_vector_length))
            input_list = vae_definition.prepare_decoder_input(latent_rep, C, S)
            decoder_outputs = decoder.predict(input_list, batch_size=batch_size, verbose=False)
            y, i, v, d, n = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

            Y.extend(y)
            I.extend(i)
            V.extend(v)
            D.extend(d)

        Y = np.asarray(Y)
        D = np.asarray(D)
        V = np.asarray(V)

        data_class.draw_pianoroll(prepare_for_drawing(Y, V), name=medley_name, show=False, save_path=save_folder +medley_name)
        Y_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(Y, I, V, D)
        mf.rolls_to_midi(Y_all_programs, all_programs, save_folder, medley_name, BPM, V_all_programs, D_all_programs)
