import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def getNotes():
  notes = []

  for file in glob.glob("/content/music/*.mid"):
    #parsiramo ulazne midi fajlove(imaju ext mid)
    midi = converter.parse(file)

    print("Parsiramo %s" %file)
    #u pocetku nemamo note za obradjivanje
    notesToParse = None
    #pokusavamo da razvrstamo po instrumentima 
    try:
      instruments = instrument.partitionByInstrument(midi)
      notesToParse = instruments.parts[0].recurse()
    except:
      notesToParse = midi.flat.notes

    #posto od tipova ulaza imamo note i akorde, moraju oba slucaja da se obrade
    #tj proveravamo sta je od ta dva u pitanju i dodajemo u []
    for el in notesToParse:
      if isinstance(el, note.Note):
        notes.append(str(el.pitch))
      #akorde predstavljamo kao nota.nota.nota......
      if isinstance(el, chord.Chord):
        notes.append('.'.join(str(i) for i in el.normalOrder))

    #serijalizujemo dobijene note i upisujemo u fajl da bismo kasnije mogli da
    #ih prenosimo i koristimo....
  with open('notes', 'wb') as fpath:
      pickle.dump(notes, fpath)

  print(len(notes))
  return notes


def prepareSequences(notes, nDiff):

  #da bitmo predvideli koja nota/koji akord je na redu, koristimo prethodnih 100 
  ##########probati sa razl vrednostima
  sequenceLength = 100

  #zelimo da izdvojimo sve tonove koji su se javljali u nasim "uzorcima"
  #tj kompozicijama koje smo citali
  pitchNames = sorted(set(i for i in notes))


  #sada sve ucitane note zelimo da napravimo preslikavanje, tj da ih predstavimo
  #kao parove str, int 
  #to kasnije mozemo iskoristili da primenimo gradijentni spust(u lstm)
  noteToInt = dict((note, number) for number, note in enumerate(pitchNames))

  networkInput = []
  networkOutput = []

  n = len(notes)

  #izlaz ce biti prva nota ili akord koji dolaze nakon odgovarajuce ulazne 
  #sekvence 

  for i in range(0, n - sequenceLength):
    sequenceIn = notes[i:i + sequenceLength]
    sequenceOut = notes[i + sequenceLength]
    networkInput.append([noteToInt[note] for note in sequenceIn])
    networkOutput.append(noteToInt[sequenceOut])

  nPatterns = len(networkInput)

  # ulaz predstavljamo u formatu kompatibilnom sa lstm slojevima
  networkInput = numpy.reshape(networkInput, (nPatterns, sequenceLength, 1))
  # normalizujemo ulaz (delimo sa brojem razlicitih nota/akorda)
  networkInput = networkInput / float(nDiff)

  networkOutput = np_utils.to_categorical(networkOutput)

  return (networkInput, networkOutput)


def create_network(networkInput, nDiff):
    """ create the structure of the neural network """
    #LSTM(Long Short Trem Memory) je sloj RRN koja prima sekvencu ulaza i vraca
    #sekvencu ili matricu (u ovom slucaju sekvencu)
    #aktivacioni sloj odredjuje koju ce aktivacionu fju nasa mreza koristiti za
    #izdracunavanje 
     #za LSTM, Dense i Activation slojeve prvi parametar je broj cvorova u njima
    #dropout parametar predstavlja koliki deo ulaznih vrednosti ce biti odbacen
    #prilikom treniranja
    # input_shape daje do znjanja mrezi kakvog ce oblika biti podaci koje ce 
    #trenirati 


    ######treba se igrati malo sa ovim slojevima 
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(networkInput.shape[1], networkInput.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(nDiff))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #poslednji sloj mreze ima isti broj cvorova kao nas izlaz da bi se direktno
    #mapiralo
    return model


def train(model, networkInput, networkOutput):

  #####ovaj deo treba objasniti prica je o gubicima
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacksList = [checkpoint]

    model.fit(networkInput, networkOutput, epochs=200, batch_size=128, callbacks=callbacksList)


def trainNetwork():
    #u ovoj funkciji treniramo mrezu
    notes = getNotes()

    # broj svih tonova(bez duplikata)
    nDiff = len(set(notes))

    networkInput, networkOutput = prepareSequences(notes, nDiff)

    model = createNetwork(networkInput, nDiff)

    train(model, networkInput, networkOutput)



